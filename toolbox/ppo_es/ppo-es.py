import logging

from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG, \
    warn_about_bad_reward_scales
from ray.tune.registry import _global_registry, ENV_CREATOR
from ray.tune.util import merge_dicts

from toolbox import initialize_ray
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.modified_rllib.multi_gpu_optimizer import \
    make_policy_optimizer_basic_modification

logger = logging.getLogger(__name__)

ppo_es_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        update_steps=100000,
    )
)


def after_train_result(trainer, result):
    warn_about_bad_reward_scales(trainer, result)  # original function

    if not hasattr(trainer, "update_policy_counter"):
        trainer.update_policy_counter = 1

    rewards = result['policy_reward_mean']
    steps = result['info']['num_steps_trained']

    update_steps = trainer.config['update_steps']
    if steps > update_steps * trainer.update_policy_counter:
        best_agent = max(rewards, key=lambda x: rewards[x])
        weights = trainer.get_policy(best_agent).get_weights()

        def _spawn_policy(policy, _):
            policy.set_weights(weights)

        # set to policies on local worker. Then all polices would be the same.
        trainer.workers.local_worker().foreach_policy(_spawn_policy)

        msg = "Current num_steps_trained is {}, exceed last update steps {}" \
              " (our update interval is {}). Current best agent is <{}> " \
              "with reward {:.4f}. We spawn it to others: {}.".format(
            steps, trainer.update_policy_counter * update_steps, update_steps,
            best_agent, rewards[best_agent], rewards
        )
        print(msg)
        logger.info(msg)
        trainer.update_policy_counter += 1

def validate_config(config):
    """
    You need to set
        config['env'] = MultiAgentEnvWrapper
        config['env_config'] = {
            'num_agents': 3,
            'env_name': 'BipedalWalker-v2'
        }
    So that this function will setup a multiagent environment for you.
    """
    assert _global_registry.contains(ENV_CREATOR, config["env"])
    env_creator = _global_registry.get(ENV_CREATOR, config["env"])
    tmp_env = env_creator(config["env_config"])
    config["multiagent"]["policies"] = {
        i: (None, tmp_env.observation_space, tmp_env.action_space, {})
        for i in tmp_env.agent_ids
    }
    config["multiagent"]["policy_mapping_fn"] = lambda x: x


PPOESTrainer = PPOTrainer.with_updates(
    name="PPOES",
    default_config=ppo_es_default_config,
    validate_config=validate_config,
    make_policy_optimizer=make_policy_optimizer_basic_modification,
    after_train_result=after_train_result
)

if __name__ == '__main__':
    # def test_train_ipd(local_mode=False):
    initialize_ray(test_mode=True, local_mode=True)
    env_name = "CartPole-v0"
    num_agents = 3

    config = {
        "num_sgd_iter": 2,
        "train_batch_size": 400,
        "env": MultiAgentEnvWrapper,
        "env_config": {"env_name": env_name, "num_agents": num_agents},
        "update_steps": 1000
    }
    tune.run(
        PPOESTrainer,
        name="DELETEME_TEST",
        verbose=2,
        stop={"timesteps_total": 10000},
        config=config
    )
