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
    validate_config_basic

logger = logging.getLogger(__name__)

smart_adaptive_extra_loss_ppo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        # How many iteration to wait if max_episode_reward is not increased.
        waiting_iteration=50,
        novelty_loss_param_step=0.05,
        joint_dataset_sample_batch_size=200,
        novelty_mode="mean",
        use_joint_dataset=True
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
        self.reward_max_stat = None
        self.step = config['novelty_loss_param_step']

    def update_novelty_loss_param(self, reward_max):
        if self.reward_max_stat is None:
            # lazy initialize
            self.reward_max_stat = deque([reward_max], maxlen=self.maxlen)
        history_max = np.max(self.reward_max_stat)
        self.reward_max_stat.append(reward_max)

        if len(self.reward_max_stat) < self.maxlen:
            # start tuning after the queue is full.
            logger.debug(
                "Current stat length: {}".format(len(self.reward_max_stat))
            )
            return self.novelty_loss_param_val

        if (reward_max < 0.0) or (reward_max > history_max * 1.1):
            logger.info(
                "Decrease alpha. from {} to {}. reward {}, history max {}"
                "".format(
                    self.novelty_loss_param_val,
                    max(0.0, self.novelty_loss_param_val - self.step),
                    reward_max, history_max
                )
            )

            # should decrease alpha
            self.novelty_loss_param_val = max(
                0.0, self.novelty_loss_param_val - self.step
            )
        elif reward_max < history_max * 0.9:
            logger.info(
                "Increase alpha. from {} to {}. reward {}, history max {}"
                "".format(
                    self.novelty_loss_param_val,
                    min(1.0, self.novelty_loss_param_val + self.step),
                    reward_max, history_max
                )
            )

            self.novelty_loss_param_val = min(
                0.5, self.novelty_loss_param_val + self.step
            )
        self.novelty_loss_param.load(
            self.novelty_loss_param_val, session=self.get_session()
        )
        return self.novelty_loss_param_val


def after_train_result(trainer, result):
    def update(policy, policy_id):
        reward_list = result["policy_reward_max"]
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
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AddLossMixin, SmartNoveltyParamMixin
    ]
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
        test_adaptive_extra_loss_trainer2

    test_smart_adaptive_extra_loss_trainer1()
    test_adaptive_extra_loss_trainer2()
