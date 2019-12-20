from ray import tune

from toolbox import initialize_ray, get_local_dir
from toolbox.cooperative_exploration.ceppo import *
from toolbox.cooperative_exploration.cetd3 import CETD3Trainer
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.marl.test_extra_loss import _base, _get_default_test_config


def _validate_base(
        extra_config,
        test_mode,
        env_name,
        trainer,
        stop=50000,
        name="DELETEME_TEST",
        num_gpus=0
):
    initialize_ray(test_mode=test_mode, local_mode=True, num_gpus=num_gpus)
    num_agents = 3
    # policy_names = ["agent{}".format(i) for i in range(num_agents)]
    env_config = {"env_name": env_name, "num_agents": num_agents}
    # env = MultiAgentEnvWrapper(env_config)
    config = {
        "seed": 0,
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        # "multiagent": {
        # "policies": {
        #     i: (None, env.observation_space, env.action_space, {})
        #     for i in policy_names
        # },
        # "policy_mapping_fn": lambda x: x,
        # },
    }
    if extra_config:
        config.update(extra_config)
    return tune.run(
        trainer,
        name=name,
        stop={"info/num_steps_sampled": stop},
        config=config
    )


def test_ceppo(local_mode=False):
    _base(
        CEPPOTrainer,
        local_mode,
        extra_config={
            "mode": tune.grid_search(
                [
                    # DISABLE,
                    # DISABLE_AND_EXPAND,
                    REPLAY_VALUES,
                    # NO_REPLAY_VALUES,
                    # DIVERSITY_ENCOURAGING,
                    # DIVERSITY_ENCOURAGING_NO_RV,
                    # DIVERSITY_ENCOURAGING_DISABLE,
                    # DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND, CURIOSITY,
                    # CURIOSITY_NO_RV,
                    # CURIOSITY_DISABLE,
                    # CURIOSITY_DISABLE_AND_EXPAND,
                    # CURIOSITY_KL,
                    # CURIOSITY_KL_NO_RV,
                    # CURIOSITY_KL_DISABLE,
                    # CURIOSITY_KL_DISABLE_AND_EXPAND
                ]
            ),
            "num_cpus_per_worker": 0.5,
            "num_workers": 1,
        },
        # extra_config={"mode": DIVERSITY_ENCOURAGING},
        env_name="Pendulum-v0",
        t=10000
    )


def test_multiple_num_agents(local_mode=False):
    num_gpus = 0
    initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=num_gpus)
    config = _get_default_test_config(
        tune.grid_search([2, 3, 4]), "BipedalWalker-v2", num_gpus
    )
    return tune.run(
        CEPPOTrainer,
        local_dir=get_local_dir(),
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        stop={"timesteps_total": 5000},
        config=config
    )


def test_single_agent(local_mode=False):
    _base(CEPPOTrainer, local_mode, dict(mode=DISABLE), num_agents=1)


def validate_ceppo():
    _validate_base(
        # {"mode": tune.grid_search(OPTIONAL_MODES)}, False, "CartPole-v0",
        {
            "mode": tune.grid_search(
                # [DISABLE, DISABLE_AND_EXPAND, REPLAY_VALUES, NO_REPLAY_VALUES]
                [REPLAY_VALUES]
            )
        },
        True,
        "CartPole-v0",
        CEPPOTrainer
    )


def test_cetd3(local_mode=False):
    num_gpus = 0
    initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=num_gpus)
    config = _get_default_test_config(
        num_agents=3, env_name="BipedalWalker-v2", num_gpus=num_gpus
    )
    if "num_sgd_iter" in config:
        config.pop("num_sgd_iter")
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


def validate_cetd3(num_gpus=0):
    from toolbox.cooperative_exploration.cetd3 import SHARE_SAMPLE, DISABLE
    _validate_base(
        {
            "mode": tune.grid_search([SHARE_SAMPLE, DISABLE]),
            "swap_prob": tune.grid_search([0.1, 0.25, 0.5]),
            "actor_hiddens": [32, 64],
            "critic_hiddens": [64, 64],
        },
        False,
        "MountainCarContinuous-v0",
        CETD3Trainer,
        num_gpus=num_gpus
    )


if __name__ == '__main__':
    # test_multiple_num_agents(local_mode=False)
    test_ceppo(local_mode=False)
    # validate_ceppo()
    # test_single_agent()
    # test_cetd3(local_mode=True)
    # validate_cetd3()
    # test_deceppo()
    # _base(
    #     CEPPOTrainer,
    #     True,
    #     extra_config={"mode": "disable"},
    #     env_name="CartPole-v0"
    # )
