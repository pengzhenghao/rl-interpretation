import logging
from collections import deque

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import update_kl
from ray.rllib.agents.ppo.ppo_policy import setup_mixins

from toolbox.marl.extra_loss_ppo_trainer import ExtraLossPPOTFPolicy, \
    ExtraLossPPOTrainer, AddLossMixin, DEFAULT_CONFIG, merge_dicts, \
    validate_config_basic, kl_and_loss_stats_modified, mixin_list

logger = logging.getLogger(__name__)

adaptive_extra_loss_ppo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        novelty_loss_param_init=0.000001,
        novelty_loss_increment=10.0,
        novelty_loss_running_length=10,
        joint_dataset_sample_batch_size=200,
        novelty_mode="mean",
        use_joint_dataset=True
    )
)


class NoveltyParamMixin(object):
    def __init__(self, config):
        # may be need to tune this value
        self.increment = config['novelty_loss_increment']

        self.novelty_loss_param_val = config['novelty_loss_param_init']
        self.novelty_target = np.nan
        # For stat purpose only.
        self.novelty_target_tensor = tf.get_variable(
            initializer=tf.constant_initializer(self.novelty_target),
            name="novelty_target",
            shape=(),
            trainable=False,
            dtype=tf.float32
        )
        self.novelty_loss_param = tf.get_variable(
            initializer=tf.constant_initializer(self.novelty_loss_param_val),
            name="novelty_loss_param",
            shape=(),
            trainable=False,
            dtype=tf.float32
        )
        self.maxlen = config['novelty_loss_running_length']
        self.novelty_stat = deque(maxlen=self.maxlen)

    def update_novelty(self, sampled_novelty):
        sampled_novelty = -sampled_novelty
        assert sampled_novelty > 0

        self.novelty_stat.append(sampled_novelty)
        if len(self.novelty_stat) < self.maxlen:
            # start tuning after the queue is full.
            logger.debug(
                "Current stat length: {}".format(len(self.novelty_stat))
            )
            return self.novelty_loss_param_val
        elif np.isnan(self.novelty_target):
            self.novelty_target = max(
                np.mean(self.novelty_stat) - self.increment, self.increment
            )
            assert self.novelty_target > 0
            logger.debug(
                "After {} iterations, we set novelty_target to {}"
                "".format(len(self.novelty_stat), self.novelty_target))

        running_mean = np.mean(self.novelty_stat)
        logger.debug(
            "Current novelty {}, mean {}, target {}, param {}".format(
                sampled_novelty, running_mean, self.novelty_target,
                self.novelty_loss_param_val
            )
        )
        if running_mean > self.novelty_target + self.increment:
            msg = "We detected the novelty {} has exceeded" \
                  " the target {}, so we increase the target" \
                  " to: {}. (instant novelty: {})".format(
                running_mean, self.novelty_target,
                self.novelty_target + self.increment,
                sampled_novelty)
            logger.info(msg)
            self.novelty_target += self.increment

        if sampled_novelty > self.novelty_target:
            self.novelty_loss_param_val *= 0.9
        elif sampled_novelty < self.novelty_target:
            self.novelty_loss_param_val = min(
                self.novelty_loss_param_val * 1.1, 1.0
            )
        self.novelty_loss_param.load(
            self.novelty_loss_param_val, session=self.get_session()
        )
        self.novelty_target_tensor.load(
            self.novelty_target, session=self.get_session()
        )
        return self.novelty_loss_param_val


def update_novelty(trainer, fetches):
    if "novelty_loss" in fetches:
        # single-agent
        trainer.workers.local_worker(
        ).for_policy(lambda pi: pi.update_novelty(fetches["novelty_loss"]))
    else:

        def update(pi, pi_id):
            if pi_id in fetches:
                pi.update_novelty(fetches[pi_id]["novelty_loss"])
            else:
                logger.debug(
                    "No data for {}, not updating"
                    " novelty_loss_param".format(pi_id)
                )

        # multi-agent
        trainer.workers.local_worker().foreach_trainable_policy(update)


def wrap_after_train_result(trainer, fetches):
    update_novelty(trainer, fetches)
    update_kl(trainer, fetches)


def setup_mixins_modified(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    AddLossMixin.__init__(policy, config)
    NoveltyParamMixin.__init__(policy, config)


def wrap_stats_fn(policy, train_batch):
    ret = kl_and_loss_stats_modified(policy, train_batch)
    ret.update(
        novelty_loss_param=policy.novelty_loss_param,
        novelty_target=policy.novelty_target_tensor
    )
    return ret


AdaptiveExtraLossPPOTFPolicy = ExtraLossPPOTFPolicy.with_updates(
    name="AdaptiveExtraLossPPOTFPolicy",
    get_default_config=lambda: adaptive_extra_loss_ppo_default_config,
    before_loss_init=setup_mixins_modified,
    stats_fn=wrap_stats_fn,
    mixins=mixin_list + [AddLossMixin, NoveltyParamMixin]
)

AdaptiveExtraLossPPOTrainer = ExtraLossPPOTrainer.with_updates(
    name="AdaptiveExtraLossPPO",
    after_optimizer_step=wrap_after_train_result,
    validate_config=validate_config_basic,
    default_config=adaptive_extra_loss_ppo_default_config,
    default_policy=AdaptiveExtraLossPPOTFPolicy,
)

if __name__ == '__main__':
    from toolbox import initialize_ray

    print("Prepare to create AELPPO")
    initialize_ray(test_mode=True, num_gpus=0)
    AdaptiveExtraLossPPOTrainer(env="BipedalWalker-v2", config=None)
