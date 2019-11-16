import logging
from collections import deque

import numpy as np
import tensorflow as tf
from ray import tune
from ray.rllib.agents.ppo.ppo import update_kl

from toolbox.marl import MultiAgentEnvWrapper, on_train_result
from toolbox.marl.extra_loss_ppo_trainer import ExtraLossPPOTFPolicy, \
    ExtraLossPPOTrainer, ValueNetworkMixin, KLCoeffMixin, AddLossMixin, \
    LearningRateSchedule, EntropyCoeffSchedule, DEFAULT_CONFIG, merge_dicts, \
    validate_config_basic, ppo_surrogate_loss, novelty_loss, \
    kl_and_loss_stats_modified
from toolbox.utils import get_local_dir, initialize_ray

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
        self.novelty_stat = None

    def update_novelty(self, sampled_novelty):
        sampled_novelty = -sampled_novelty

        if self.novelty_stat is None:
            # lazy initialize
            self.novelty_target = min(
                sampled_novelty - self.increment, self.increment
            )
            self.novelty_stat = deque(
                [self.novelty_target] * self.maxlen,
                maxlen=self.maxlen
            )

        self.novelty_stat.append(sampled_novelty)
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

        if sampled_novelty > self.increment + self.novelty_target:
            # if sampled_novelty > 2.0 * self.novelty_target:
            self.novelty_loss_param_val *= 0.5
        elif sampled_novelty < self.novelty_target - self.increment:
            # elif sampled_novelty < 0.5 * self.novelty_target:
            self.novelty_loss_param_val *= 1.5
        self.novelty_loss_param.load(self.novelty_loss_param_val,
                                     session=self.get_session())
        self.novelty_target_tensor.load(self.novelty_target,
                                        session=self.get_session())
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


def adaptive_extra_loss_ppo_loss(policy, model, dist_class, train_batch):
    """Add novelty loss with original ppo loss"""
    original_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    nov_loss = novelty_loss(policy, model, dist_class, train_batch)
    alpha = policy.novelty_loss_param
    total_loss = (1 - alpha) * original_loss + alpha * nov_loss
    policy.total_loss = total_loss
    return total_loss


def wrap_after_train_result(trainer, fetches):
    update_novelty(trainer, fetches)
    update_kl(trainer, fetches)


def setup_mixins_modified(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
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
    loss_fn=adaptive_extra_loss_ppo_loss,
    stats_fn=wrap_stats_fn,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AddLossMixin, NoveltyParamMixin
    ]
)

AdaptiveExtraLossPPOTrainer = ExtraLossPPOTrainer.with_updates(
    name="AdaptiveExtraLossPPO",
    after_optimizer_step=wrap_after_train_result,
    validate_config=validate_config_basic,
    default_config=adaptive_extra_loss_ppo_default_config,
    default_policy=AdaptiveExtraLossPPOTFPolicy,
)


def test1(extra_config=None):
    num_agents = 3
    num_gpus = 0

    # This is only test code.
    initialize_ray(test_mode=True, local_mode=True, num_gpus=num_gpus)

    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]

    env_config = {"env_name": "BipedalWalker-v2", "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "num_gpus": num_gpus,
        "log_level": "DEBUG",
        "joint_dataset_sample_batch_size": 37,
        "multiagent": {
            "policies": {
                i: (None, env.observation_space, env.action_space, {})
                for i in policy_names
            },
            "policy_mapping_fn": lambda x: x,
        },
        "callbacks": {
            "on_train_result": on_train_result
        },
    }
    if extra_config:
        config.update(extra_config)

    tune.run(
        AdaptiveExtraLossPPOTrainer,
        local_dir=get_local_dir(),
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        stop={"timesteps_total": 50000},
        config=config
    )


def test2():
    test1({"use_joint_dataset": False})


if __name__ == '__main__':
    test1()
    # test2()
