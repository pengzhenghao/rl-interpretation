import logging
from collections import deque

import tensorflow as tf
from ray import tune
from ray.rllib.agents.ppo.ppo import update_kl

from toolbox.marl import MultiAgentEnvWrapper, on_train_result
from toolbox.marl.extra_loss_ppo_trainer import ExtraLossPPOTFPolicy, \
    ExtraLossPPOTrainer, ValueNetworkMixin, KLCoeffMixin, AddLossMixin, \
    LearningRateSchedule, EntropyCoeffSchedule
from toolbox.utils import get_local_dir, initialize_ray

logger = logging.getLogger(__name__)

import numpy as np


class NoveltyParamMixin(object):
    def __init__(self, config):
        self.novelty_loss_param_val = config["novelty_loss_param"]
        self.increment = 10.0
        self.novelty_loss_param_target = self.increment  # FIXME should from
        # config
        self.novelty_loss_param = tf.get_variable(
            initializer=tf.constant_initializer(self.novelty_loss_param_val),
            name="novelty_loss_param",
            shape=(),
            trainable=False,
            dtype=tf.float32)

        maxlen = 100
        self.novelty_stat = deque([0] * maxlen, maxlen=maxlen)

    def update_novelty(self, sampled_novelty):
        sampled_novelty = -sampled_novelty
        self.novelty_stat.append(sampled_novelty)
        running_mean = np.mean(self.novelty_stat)
        if running_mean > self.novelty_loss_param_target + self.increment:
            msg = "We detected the novelty {} has exceeded" \
                  " the target {}, so we increase the target" \
                  " to: {}. (instant novelty: {})".format(
                running_mean, self.novelty_loss_param_target,
                self.novelty_loss_param_target + self.increment,
                sampled_novelty)
            logger.info(msg)
            print(msg)
            self.novelty_loss_param_target += self.increment

        if sampled_novelty > 2.0 * self.novelty_loss_param_target:
            self.novelty_loss_param_val *= 1.5
        elif sampled_novelty < 0.5 * self.novelty_loss_param_target:
            self.novelty_loss_param_val *= 0.5
        self.novelty_loss_param.load(self.novelty_loss_param_val,
                                     session=self.get_session())
        return self.novelty_loss_param_val


def update_novelty(trainer, fetches):
    if "novelty_loss" in fetches:
        # single-agent
        trainer.workers.local_worker().for_policy(
            lambda pi: pi.update_novelty(fetches["novelty_loss"]))
    else:

        def update(pi, pi_id):
            if pi_id in fetches:
                pi.update_kl(fetches[pi_id]["novelty_loss"])
            else:
                logger.debug("No data for {}, not updating"
                             " novelty_loss_param".format(pi_id))

        # multi-agent
        trainer.workers.local_worker().foreach_trainable_policy(update)


def warp_after_train_result(trainer, fetches):
    update_novelty(trainer, fetches)
    update_kl(trainer, fetches)


def setup_mixins_modified(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    NoveltyParamMixin.__init__(policy, config)


AdaptiveExtraLossPPOTFPolicy = ExtraLossPPOTFPolicy.with_updates(
    name="AdaptiveExtraLossPPOTFPolicy",
    # get_default_config=lambda: extra_loss_ppo_default_config,
    # postprocess_fn=postprocess_ppo_gae,
    # stats_fn=kl_and_loss_stats_modified,
    # loss_fn=extra_loss_ppo_loss,
    before_loss_init=setup_mixins_modified,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AddLossMixin, NoveltyParamMixin
        # KLCoeffMixinModified
    ]
)

AdaptiveExtraLossPPOTrainer = ExtraLossPPOTrainer.with_updates(
    name="AdaptiveExtraLossPPO",

    after_optimizer_step=warp_after_train_result,
    # default_config=extra_loss_ppo_default_config,
    # validate_config=validate_config_modified,
    default_policy=AdaptiveExtraLossPPOTFPolicy,
    # mixin=[]
    # make_policy_optimizer=choose_policy_optimizer
)


def test1(extra_config=None):
    num_agents = 3
    num_gpus = 0

    # This is only test code.
    initialize_ray(test_mode=True, local_mode=False, num_gpus=num_gpus)

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
        stop={"timesteps_total": 10000},
        config=config
    )


def test2():
    test1({"use_joint_dataset": False})


if __name__ == '__main__':
    test1()
    test2()
