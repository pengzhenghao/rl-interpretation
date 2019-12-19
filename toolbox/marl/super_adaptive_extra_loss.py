"""We take idea from paper 'Improving Exploration in Evolution Strategies
for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents'
by annealing the alpha to balance novelty loss and policy loss"""

import logging
from collections import deque

import numpy as np
import tensorflow as tf

from toolbox.marl.extra_loss_ppo_trainer import ExtraLossPPOTrainer, \
    ExtraLossPPOTFPolicy, merge_dicts, DEFAULT_CONFIG, \
    kl_and_loss_stats_modified, ValueNetworkMixin, KLCoeffMixin, \
    EntropyCoeffSchedule, LearningRateSchedule, AddLossMixin, \
    validate_config_basic, mixin_list
from toolbox.marl.smart_adaptive_extra_loss import SmartAdaptiveExtraLossPPOTrainer, SmartAdaptiveExtraLossPPOTFPolicy
from toolbox.marl.adaptive_extra_loss import wrap_stats_fn

logger = logging.getLogger(__name__)

pappo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        novelty_loss_param_init=0.000001,
        novelty_loss_param_step=0.005,
        novelty_loss_increment=10.0,
        novelty_loss_running_length=10,
        waiting_iteration=50,
        joint_dataset_sample_batch_size=200,
        novelty_mode="mean",
        performance_evaluation_metric="max",
        use_joint_dataset=True,
        performance_lower_bound=0.9,
        performance_upper_bound=1.0
    )
)


class SmartNoveltyParamMixin(object):
    def __init__(self, config):

        # may be need to tune this value
        self.increment = config['novelty_loss_increment']

        self.novelty_target = np.nan
        # For stat purpose only.
        self.novelty_target_tensor = tf.get_variable(
            initializer=tf.constant_initializer(self.novelty_target),
            name="novelty_target",
            shape=(),
            trainable=False,
            dtype=tf.float32
        )

        self.novelty_loss_param_val = config['novelty_loss_param_init']
        self.novelty_loss_param = tf.get_variable(
            initializer=tf.constant_initializer(self.novelty_loss_param_val),
            name="novelty_loss_param",
            shape=(),
            trainable=False,
            dtype=tf.float32
        )

        self.maxlen = config['waiting_iteration']
        self.reward_stat = None
        self.novelty_stat = None
        self.step = config['novelty_loss_param_step']
        self.config = config
        if config['performance_evaluation_metric'] == "max":
            self.metric = np.min
        elif config['performance_evaluation_metric'] == "mean":
            self.metric = np.mean
        else:
            raise ValueError(
                "We expect 'performance_evaluation_metric' "
                "in ['max', 'mean']"
            )

    def update_novelty_loss_param(self, performance, sampled_novelty):
        assert performance is not None
        assert performance != float('nan')
        assert sampled_novelty < 0

        sampled_novelty = -sampled_novelty

        if self.reward_stat is None:
            # lazy initialize
            self.reward_stat = deque([performance], maxlen=self.maxlen)
            self.novelty_target = min(
                sampled_novelty - self.increment, self.increment
            )
            self.novelty_stat = deque(
                [self.novelty_target] * self.maxlen, maxlen=self.maxlen
            )

        self.novelty_stat.append(sampled_novelty)
        running_mean = np.mean(self.novelty_stat)
        logger.debug(
            "Current novelty {}, mean {}, target {}, param {}".format(
                sampled_novelty, running_mean, self.novelty_target,
                self.novelty_loss_param_val
            )
        )

        # SAEL
        history_performance = self.metric(self.reward_stat)
        self.reward_stat.append(performance)
        if len(self.reward_stat) < self.maxlen:
            # start tuning after the queue is full.
            logger.debug(
                "Current stat length: {}".format(len(self.reward_stat))
            )
            # TODO return what?
            return self.novelty_loss_param_val

        if (performance < 0.0) or (performance > history_performance):

            msg = "We detected the performance is stuck, " \
                  "so we increase the target to: {}. (current performance: " \
                  "{}, history performance {})".format(
                self.novelty_target, self.novelty_target+self.increment,
                performance, history_performance)
            logger.info(msg)

            self.novelty_target += self.increment

            # should decrease alpha

            # old_value = self.novelty_loss_param_val
            # self.novelty_loss_param_val = max(
            #     0.0, self.novelty_loss_param_val - self.step
            # )
            # logger.info(
            #     "Decrease alpha. from {} to {}. reward {}, history max {}"
            #     "".format(
            #         old_value, self.novelty_loss_param_val, performance,
            #         history_performance
            #     )
            # )

        elif performance < history_performance * 0.9:
            pass

            # old_value = self.novelty_loss_param_val
            # self.novelty_loss_param_val = min(
            #     0.5, self.novelty_loss_param_val + self.step
            # )
            # logger.info(
            #     "Increase alpha. from {} to {}. reward {}, history max {}"
            #     "".format(
            #         old_value, self.novelty_loss_param_val, performance,
            #         history_performance
            #     )
            # )

        # AEL
        if running_mean > self.novelty_target + self.increment:
            msg = "We detected the novelty {} has exceeded" \
                  " the target {}, so we increase the target" \
                  " to: {}. (instant novelty: {})".format(
                running_mean, self.novelty_target,
                self.novelty_target + self.increment,
                sampled_novelty)
            logger.info(msg)
            self.novelty_target += self.increment
        if sampled_novelty > self.increment + self.novelty_target:
            # if sampled_novelty > 2.0 * self.novelty_target:
            self.novelty_loss_param_val *= 0.5
        elif sampled_novelty < self.novelty_target - self.increment:
            # elif sampled_novelty < 0.5 * self.novelty_target:
            self.novelty_loss_param_val *= 1.5
        self.novelty_loss_param.load(
            self.novelty_loss_param_val, session=self.get_session()
        )
        self.novelty_target_tensor.load(
            self.novelty_target, session=self.get_session()
        )

        self.novelty_loss_param.load(
            self.novelty_loss_param_val, session=self.get_session()
        )
        return self.novelty_loss_param_val


def after_train_result(trainer, result):
    def update(policy, policy_id):
        if policy.config['performance_evaluation_metric'] == "max":
            reward_list = result["policy_reward_max"]
        elif policy.config['performance_evaluation_metric'] == "mean":
            reward_list = result["policy_reward_mean"]
        else:
            raise ValueError("performance_evaluation_metric wrong!")

        if policy_id in reward_list:
            policy.update_novelty_loss_param(reward_list[policy_id])
        else:
            logger.debug(
                "No policy_reward_max for {}, not updating "
                "novelty_loss_param.".format(policy_id)
            )

    trainer.workers.local_worker().foreach_trainable_policy(update)


def setup_mixins_modified(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    SmartNoveltyParamMixin.__init__(policy, config)


PAPPOTFPolicy = SmartAdaptiveExtraLossPPOTFPolicy.with_updates(
    name="PAPPOTFPolicy",
    get_default_config=lambda: pappo_default_config,
    # before_loss_init=setup_mixins_modified,
    stats_fn=wrap_stats_fn,
    # mixins=mixin_list + [AddLossMixin, SmartNoveltyParamMixin]
)

PAPPOTrainer = SmartAdaptiveExtraLossPPOTrainer.with_updates(
    name="PAPPO",
    default_config=pappo_default_config,
    default_policy=PAPPOTFPolicy,
    # after_train_result=after_train_result,
    # validate_config=validate_config_basic,
)

if __name__ == '__main__':
    from toolbox.marl.test_extra_loss import \
        test_smart_adaptive_extra_loss_trainer1, \
        test_smart_adaptive_extra_loss_trainer2, \
        test_smart_adaptive_extra_loss_trainer3

    test_smart_adaptive_extra_loss_trainer1(False)
    test_smart_adaptive_extra_loss_trainer2(False)
    test_smart_adaptive_extra_loss_trainer3(False)
