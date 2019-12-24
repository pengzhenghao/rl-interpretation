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

logger = logging.getLogger(__name__)

smart_adaptive_extra_loss_ppo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        # How many iteration to wait if max_episode_reward is not increased.
        waiting_iteration=50,
        novelty_loss_param_step=0.005,
        joint_dataset_sample_batch_size=200,
        novelty_mode="mean",
        use_joint_dataset=True,
        performance_evaluation_metric="max",
    )
)


class SmartNoveltyParamMixin(object):
    def __init__(self, config):
        self.novelty_loss_param_val = 0.0
        self.novelty_loss_param = tf.get_variable(
            initializer=tf.constant_initializer(self.novelty_loss_param_val),
            name="novelty_loss_param",
            shape=(),
            trainable=False,
            dtype=tf.float32
        )
        self.maxlen = config['waiting_iteration']
        self.reward_stat = deque(maxlen=self.maxlen)
        self.step = config['novelty_loss_param_step']
        if config['performance_evaluation_metric'] == "max":
            self.metric = np.min
        elif config['performance_evaluation_metric'] == "mean":
            self.metric = np.mean
        else:
            raise ValueError(
                "We expect 'performance_evaluation_metric' "
                "in ['max', 'mean']"
            )

    def update_novelty_loss_param(self, performance):
        history_performance = self.metric(self.reward_stat)
        self.reward_stat.append(performance)

        if len(self.reward_stat) < self.maxlen:
            # start tuning after the queue is full.
            logger.debug(
                "Current stat length: {}".format(len(self.reward_stat))
            )
            return self.novelty_loss_param_val

        if (performance < 0.0) or (performance > history_performance * 1.1):
            # should decrease alpha
            old_value = self.novelty_loss_param_val
            self.novelty_loss_param_val = max(
                0.0, self.novelty_loss_param_val - self.step
            )

            logger.info(
                "Decrease alpha. from {} to {}. reward {}, history max {}"
                "".format(
                    old_value, self.novelty_loss_param_val, performance,
                    history_performance
                )
            )

        elif performance < history_performance * 0.9:
            old_value = self.novelty_loss_param_val
            self.novelty_loss_param_val = min(
                0.5, self.novelty_loss_param_val + self.step
            )
            logger.info(
                "Increase alpha. from {} to {}. reward {}, history max {}"
                "".format(
                    old_value, self.novelty_loss_param_val, performance,
                    history_performance
                )
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


def wrap_stats_fn(policy, train_batch):
    ret = kl_and_loss_stats_modified(policy, train_batch)
    ret.update(novelty_loss_param=policy.novelty_loss_param)
    return ret


SmartAdaptiveExtraLossPPOTFPolicy = ExtraLossPPOTFPolicy.with_updates(
    name="SAELPPOTFPolicy",
    get_default_config=lambda: smart_adaptive_extra_loss_ppo_default_config,
    before_loss_init=setup_mixins_modified,
    stats_fn=wrap_stats_fn,
    mixins=mixin_list + [AddLossMixin, SmartNoveltyParamMixin]
)

SmartAdaptiveExtraLossPPOTrainer = ExtraLossPPOTrainer.with_updates(
    name="SAELPPO",
    after_train_result=after_train_result,
    validate_config=validate_config_basic,
    default_config=smart_adaptive_extra_loss_ppo_default_config,
    default_policy=SmartAdaptiveExtraLossPPOTFPolicy,
)

if __name__ == '__main__':
    from toolbox.marl.test_extra_loss import \
        test_smart_adaptive_extra_loss_trainer1, \
        test_smart_adaptive_extra_loss_trainer2, \
        test_smart_adaptive_extra_loss_trainer3

    test_smart_adaptive_extra_loss_trainer1(False)
    test_smart_adaptive_extra_loss_trainer2(False)
    test_smart_adaptive_extra_loss_trainer3(False)
