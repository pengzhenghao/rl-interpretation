import os
from collections import OrderedDict

import gym
import numpy as np
from ray import tune

from toolbox import initialize_ray
from toolbox.evaluate import restore_agent
from toolbox.process_data import read_yaml

env_config_required_items = ['env_name', 'novelty_threshold', 'yaml_path']

T_START = 20
LOWER_NOVEL_BOUND = -0.1


class IPDEnv:
    def __init__(self, env_config):
        for key in env_config_required_items:
            assert key in env_config

        self.env = gym.make(env_config['env_name'])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.prev_obs = None

        # load agents from yaml file which contains checkpoints information.
        name_ckpt_mapping = read_yaml(env_config['yaml_path'], number=2)
        extra_config = {"num_workers": 0, "num_cpus_per_worker": 0}
        self.agent_pool = OrderedDict()
        for name, ckpt in name_ckpt_mapping.items():
            assert ckpt['env_name'] == env_config['env_name']
            self.agent_pool[name] = restore_agent(
                ckpt['run_name'], ckpt['path'], ckpt['env_name'], extra_config
            )

        self.novelty_recorder = {k: 0.0 for k in self.agent_pool.keys()}
        self.novelty_recorder_count = 0
        self.novelty_sum = 0.0
        self.threshold = env_config['novelty_threshold']

    def reset(self, **kwargs):
        self.prev_obs = self.env.reset(**kwargs)
        self.novelty_recorder = {k: 0.0 for k in self.agent_pool.keys()}
        self.novelty_recorder_count = 0
        self.novelty_sum = 0.0
        return self.prev_obs

    def step(self, action):
        assert self.prev_obs is not None
        early_stop = self._criterion(action)

        o, r, original_d, i = self.env.step(action)
        self.prev_obs = o
        done = early_stop or original_d
        i['early_stop'] = done
        return o, r, done, i

    def _criterion(self, action):
        """Compute novelty, update recorder and return early-stop flag."""
        for agent_name, agent in self.agent_pool.items():
            act = agent.compute_action(self.prev_obs)
            novelty = np.linalg.norm(act - action)
            self.novelty_recorder[agent_name] += novelty
        self.novelty_recorder_count += 1

        if self.novelty_recorder_count < T_START:
            return False

        min_novelty = \
            min(self.novelty_recorder.values()) / self.novelty_recorder_count
        min_novelty = min_novelty - self.threshold

        self.novelty_sum += min_novelty
        if self.novelty_sum <= LOWER_NOVEL_BOUND:
            return True
        return False


def on_episode_end(info):
    envs = info['env'].get_unwrapped()
    novelty = np.mean([env.novelty_sum for env in envs])
    info['episode'].custom_metrics['novelty'] = novelty


def test_maddpg_custom_metrics():
    extra_config = {
        "env": IPDEnv,
        "env_config": {
            "env_name": "BipedalWalker-v2",
            "novelty_threshold": 0.0,
            "yaml_path": os.path.abspath("../data/yaml/test-2-agents.yaml")
        },
        "callbacks": {"on_episode_end": on_episode_end},
    }
    initialize_ray(test_mode=True, local_mode=False)
    tune.run("PPO", stop={"training_iteration": 10}, config=extra_config)


if __name__ == "__main__":
    test_maddpg_custom_metrics()
