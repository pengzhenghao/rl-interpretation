from ray import tune

from toolbox import initialize_ray, get_local_dir
from toolbox.cooperative_exploration.ceppo import \
    CEPPOTrainer, OPTIONAL_MODES, DISABLE
from toolbox.cooperative_exploration.ceppo_encourage_diversity import \
    DECEPPOTrainer
from toolbox.cooperative_exploration.cetd3 import CETD3Trainer
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.marl.test_extra_loss import _base, _get_default_test_config


def _validate_base(
        extra_config,
        test_mode,
        env_name,
        trainer,
        stop=50000,
        name="DELETEME_TEST_CEPPO"
):
    initialize_ray(test_mode=test_mode, local_mode=False)
    num_agents = 3
    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]
    env_config = {"env_name": env_name, "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "seed": 0,
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                i: (None, env.observation_space, env.action_space, {})
                for i in policy_names
            },
            "policy_mapping_fn": lambda x: x,
        },
    }
    if extra_config:
        config.update(extra_config)
    return tune.run(
        trainer,
        name=name,
        stop={"info/num_steps_sampled": stop},
        config=config
    )


def debug_ceppo(local_mode):
    _base(
        CEPPOTrainer,
        local_mode,
        extra_config={"mode": tune.grid_search(OPTIONAL_MODES)},
        env_name="CartPole-v0"
    )


def test_single_agent():
    _base(CEPPOTrainer, True, dict(mode=DISABLE), num_agents=1)


def test_deceppo(local_mode=False):
    _base(
        DECEPPOTrainer,
        local_mode,
        # {"mode": "replay_values"}, t=1000)
        {"mode": tune.grid_search(OPTIONAL_MODES)},
        t=1000
    )


def validate_ceppo():
    _validate_base(
        {"mode": tune.grid_search(OPTIONAL_MODES)}, False, "CartPole-v0",
        CEPPOTrainer
    )


def test_cetd3(local_mode=False):
    num_gpus = 0
    initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=num_gpus)
    config = _get_default_test_config(
        num_agents=3, env_name="BipedalWalker-v2", num_gpus=num_gpus
    )
    config.pop("sgd_minibatch_size")
    config['timesteps_per_iteration'] = 80
    config['pure_exploration_steps'] = 80
    config['learning_starts'] = 180
    tune.run(
        CETD3Trainer,
        local_dir=get_local_dir(),
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        stop={"timesteps_total": 2000},
        config=config
    )


def validate_cetd3():
    from toolbox.cooperative_exploration.cetd3 import SHARE_SAMPLE
    _validate_base(
        {"mode": tune.grid_search([SHARE_SAMPLE, None])}, False,
        "MountainCarContinuous-v0", CETD3Trainer
    )


if __name__ == '__main__':
    # debug_ceppo(local_mode=False)
    # validate_ceppo(disable=False, test_mode=False)
    # test_single_agent()
    test_cetd3(local_mode=True)
    # validate_cetd3()
    # test_deceppo()
    # _base(
    #     CEPPOTrainer,
    #     True,
    #     extra_config={"mode": "disable"},
    #     env_name="CartPole-v0"
    # )
