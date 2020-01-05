import copy
import json
import logging
import pickle

import gym
import numpy as np
import ray
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import PPOTrainer, PPOTFPolicy, DEFAULT_CONFIG

from toolbox.ipd.tnb_policy import setup_mixins_tnb, AgentPoolMixin, \
    KLCoeffMixin, EntropyCoeffSchedule, LearningRateSchedule, \
    ValueNetworkMixin, merge_dicts

logger = logging.getLogger(__name__)

I_AM_CLONE = 'i_am_clone'
T_START = 20
LOWER_NOVEL_BOUND = -0.1


def on_episode_end(info):
    envs = info['env'].get_unwrapped()
    novelty_sum = np.mean([env.novelty_sum for env in envs])
    info['episode'].custom_metrics['novelty_sum'] = novelty_sum
    info['episode'].custom_metrics['early_stop_ratio'] = np.mean(
        [
            i['early_stop']
            for i in info['episode']._agent_to_last_info.values()
        ]
    )


ipd_default_config = merge_dicts(
    DEFAULT_CONFIG,
    {
        "checkpoint_dict": "{}",
        "novelty_threshold": 0.5,

        # don't touch
        "use_novelty_value_network": False,
        "use_preoccupied_agent": False,
        "callbacks": {
            "on_episode_end": on_episode_end
        },
        "distance_mode": "min",
        "disable_tnb": False,
        I_AM_CLONE: False
    }
)


class IPDEnv(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make(env_config['env_name'])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.prev_obs = None
        self.policies_pool = {}
        self.threshold = env_config['novelty_threshold']
        self.novelty_stat = None
        self.distance_mode = "min"
        self.novelty_recorder = None
        self.novelty_recorder_count = None
        self.novelty_sum = None

    def _init(self):
        self.novelty_recorder = {k: 0.0 for k in self.policies_pool.keys()}
        self.novelty_recorder_count = 0
        self.novelty_sum = 0.0

    def reset(self, **kwargs):
        self.prev_obs = self.env.reset(**kwargs)
        self._init()
        return self.prev_obs

    def step(self, action):
        assert self.prev_obs is not None
        early_stop = self._criterion(action)
        o, r, original_d, i = self.env.step(action)
        self.prev_obs = o
        done = early_stop or original_d
        i['early_stop'] = early_stop
        return o, r, done, i

    def _criterion(self, action):
        """Compute novelty, update recorder and return early-stop flag."""
        if len(self.policies_pool) == 0:
            return False
        if self.novelty_recorder is None:
            self._init()

        for agent_name, policy in self.policies_pool.items():
            act, _, _ = policy.compute_single_action(self.prev_obs, [])
            novelty = np.linalg.norm(act - action)
            self.novelty_recorder[agent_name] += novelty
        self.novelty_recorder_count += 1
        if self.novelty_recorder_count < T_START:
            return False
        min_novelty = min(
            self.novelty_recorder.values()
        ) / self.novelty_recorder_count
        min_novelty = min_novelty - self.threshold
        self.novelty_sum += min_novelty
        if self.novelty_sum <= LOWER_NOVEL_BOUND:
            return True
        return False

    def seed(self, s=None):
        self.env.seed(s)


def _restore_state(ckpt):
    wkload = pickle.load(open(ckpt, 'rb'))['worker']
    state = pickle.loads(wkload)['state']['default_policy']
    return state


def after_init(trainer):
    if trainer.config[I_AM_CLONE]:
        return
    checkpoint_dict = json.loads(trainer.config['checkpoint_dict'])
    if len(checkpoint_dict) == 0:
        return
    weights = {
        k: _restore_state(ckpt['path']) if ckpt is not None else None
        for k, ckpt in checkpoint_dict.items()
        # if ckpt is not None
    }
    weights = ray.put(weights)

    def _init_pool(worker):
        """We load the policies pool at each worker, instead of each policy,
        to save memory."""
        local_weights = ray.get(weights)
        tmp_policy = worker.get_policy()
        policies_pool = {}
        for agent_name, agent_weight in local_weights.items():
            tmp_config = copy.deepcopy(tmp_policy.config)
            # disable the private worker of each policy, to save resource.
            tmp_config.update(
                {
                    "num_workers": 0,
                    "num_cpus_per_worker": 0,
                    "num_cpus_for_driver": 0.2,
                    "num_gpus": 0.1,
                    I_AM_CLONE: True
                }
            )
            # build the policy and restore the weights.
            with tf.variable_scope("polices_pool/" + agent_name,
                                   reuse=tf.AUTO_REUSE):
                policy = IPDPolicy(
                    tmp_policy.observation_space, tmp_policy.action_space,
                    tmp_config
                )
                if agent_weight is not None:
                    policy.set_weights(agent_weight)
            policies_pool[agent_name] = policy
        worker.policies_pool = policies_pool  # add new attribute to worker

        def _set_polices_pool(env):
            env.policies_pool = worker.policies_pool

        worker.foreach_env(_set_polices_pool)

    trainer.workers.foreach_worker(_init_pool)


IPDPolicy = PPOTFPolicy.with_updates(
    name="IPDPolicy",
    get_default_config=lambda: ipd_default_config,
    before_loss_init=setup_mixins_tnb,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AgentPoolMixin
    ]
)

IPDTrainer = PPOTrainer.with_updates(
    name="IPD",
    default_config=ipd_default_config,
    after_init=after_init,
    default_policy=IPDPolicy
)

if __name__ == '__main__':
    from ray import tune

    from toolbox import initialize_ray

    initialize_ray(test_mode=True, local_mode=False)
    env_name = "CartPole-v0"
    config = {
        "num_sgd_iter": 2,
        "env": IPDEnv,
        "env_config": {
            "env_name": env_name,
            "novelty_threshold": 0.0,
        },
        'checkpoint_dict': '{"test": null}'
    }
    tune.run(
        IPDTrainer,
        name="DELETEME_TEST",
        verbose=2,
        stop={"timesteps_total": 50000},
        config=config
    )
