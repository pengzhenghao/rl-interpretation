import copy
import logging
import os.path as osp
import pickle
import json
import numpy as np
import ray
import scipy.signal
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import PPOTrainer, PPOTFPolicy, DEFAULT_CONFIG, \
    validate_config as validate_config_ppo
from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae

from toolbox.ipd.tnb_policy import setup_mixins_tnb, AgentPoolMixin, \
    KLCoeffMixin, EntropyCoeffSchedule, LearningRateSchedule, \
    ValueNetworkMixin, merge_dicts
import gym
I_AM_CLONE = 'i_am_clone'
logger = logging.getLogger(__name__)


def on_episode_end(info):
    episode = info['episode']
    # print('pass')


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
        "batch_mode": "complete_episodes",
        "disable_tnb": False,
        I_AM_CLONE: False
    }
)

T_START = 20
LOWER_NOVEL_BOUND = -0.1


class IPDEnv:
    """
    A hacking workaround to implement IPD. This is not used for
    large-scale training since assign each environment with N agents
    is not practicable.
    """

    def __init__(self, env_config):
        print('env_config:', env_config)
        # exit(0)
        # assert 'yaml_path' in env_config, "Should contain yaml_path in config"
        # assert isinstance(env_config['yaml_path'], str)
        # assert osp.exists(env_config['yaml_path'])
        # for key in env_config_required_items:
        #     assert key in env_config

        self.env = gym.make(env_config['env_name'])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.prev_obs = None

        # self.policies_pool = env_config['policies_pool']

        self.policies_pool = {}

        # load agents from yaml file which contains checkpoints information.
        # name_ckpt_mapping = read_yaml(env_config['yaml_path'], number=2)
        # extra_config = {"num_workers": 0, "num_cpus_per_worker": 0}
        # self.agent_pool = OrderedDict()
        # for name, ckpt in name_ckpt_mapping.items():
        #     assert ckpt['env_name'] == env_config['env_name']
        #     self.agent_pool[name] = restore_agent(
        #         ckpt['run_name'], ckpt['path'], ckpt['env_name'],
        #         extra_config
        #     )

        # self.novelty_recorder = {k: 0.0 for k in self.agent_pool.keys()}
        # self.novelty_recorder_count = 0
        # self.novelty_sum = 0.0
        self.threshold = env_config['novelty_threshold']

    def reset(self, **kwargs):
        self.prev_obs = self.env.reset(**kwargs)
        # self.novelty_recorder = {k: 0.0 for k in self.agent_pool.keys()}
        # self.novelty_recorder_count = 0
        # self.novelty_sum = 0.0
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
        # for agent_name, agent in self.agent_pool.items():
        #     act = agent.compute_action(self.prev_obs)
        #     novelty = np.linalg.norm(act - action)
        #     self.novelty_recorder[agent_name] += novelty
        # self.novelty_recorder_count += 1
        #
        # if self.novelty_recorder_count < T_START:
        #     return False
        #
        # min_novelty = \
        #     min(self.novelty_recorder.values()) / self.novelty_recorder_count
        # min_novelty = min_novelty - self.threshold
        #
        # self.novelty_sum += min_novelty
        # if self.novelty_sum <= LOWER_NOVEL_BOUND:
        #     return True
        return False

    def seed(self, s):
        self.env.seed(s)


def on_episode_end(info):
    envs = info['env'].get_unwrapped()
    novelty = np.mean([env.novelty_sum for env in envs])
    info['episode'].custom_metrics['novelty'] = novelty


# def test_maddpg_custom_metrics():
#     extra_config = {
#         "env": IPDEnv,
#         "env_config": {
#             "env_name": "BipedalWalker-v2",
#             "novelty_threshold": 0.0,
#             "checkpoint_dict": os.path.abspath(
#             "../data/yaml/test-2-agents.yaml")
#         },
#         "callbacks": {
#             "on_episode_end": on_episode_end
#         },
#     }
#     initialize_ray(test_mode=True, local_mode=False)
#     tune.run("PPO", stop={"training_iteration": 10}, config=extra_config)


def _restore_state(ckpt):
    wkload = pickle.load(open(ckpt, 'rb'))['worker']
    state = pickle.loads(wkload)['state']['default_policy']
    return state


def after_init(trainer):
    if trainer.config[I_AM_CLONE]:
        return

    checkpoint_dict = json.loads(trainer.config['checkpoint_dict'])

    weights = {
        k: _restore_state(ckpt) if ckpt is not None else None
        for k, ckpt in checkpoint_dict.items()
        # if ckpt is not None
    }

    if len(weights) == 0:
        return

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
    print('finish!')



def accumulate(x, gamma=1.0):
    return scipy.signal.lfilter([1], [1, -gamma], x, axis=0)


def postprocess_fn(policy, batch, others_batches, episode):
    # clip the episode.
    if policy.initialized_policies_pool:
        nov = policy.compute_novelty(batch)

        criterion = accumulate(nov)
        index = np.argmax(criterion <= 0)
        batch = batch.slice(0, index)

        episode.custom_metrics['novelty_min'] = nov.min()
        episode.custom_metrics['novelty_mean'] = nov.mean()
        episode.custom_metrics['novelty_max'] = nov.max()
        episode.custom_metrics['keep_ratio'] = index / len(nov)
        episode.custom_metrics['keep_length'] = index
        episode.custom_metrics['original_length'] = len(nov)
    if batch.count > 0:
        batch = postprocess_ppo_gae(policy, batch, others_batches, episode)
    return batch


IPDPolicy = PPOTFPolicy.with_updates(
    name="IPDPolicy",
    get_default_config=lambda: ipd_default_config,
    before_loss_init=setup_mixins_tnb,
    postprocess_fn=postprocess_fn,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AgentPoolMixin
    ]
)


def validate_config(config):
    validate_config_ppo(config)

    config['env_config'][
        'policies_pool'] = {}  # wait for fill by setup_policies_pool



IPDTrainer = PPOTrainer.with_updates(
    name="IPD",
    default_config=ipd_default_config,
    # before_init=setup_policies_pool,
    after_init=after_init,
    default_policy=IPDPolicy,
    validate_config=validate_config,
)

if __name__ == '__main__':
    from ray import tune

    from toolbox import initialize_ray

    initialize_ray(test_mode=True, local_mode=True)
    env_name = "CartPole-v0"
    config = {"num_sgd_iter": 2, "env": IPDEnv,
              "env_config": {
                  "env_name": env_name,
                  "novelty_threshold": 0.0,
              },
              'checkpoint_dict': '{"test": null}'}
    tune.run(
        IPDTrainer,
        name="DELETEME_TEST",
        verbose=2,
        stop={"timesteps_total": 50000},
        config=config
    )
