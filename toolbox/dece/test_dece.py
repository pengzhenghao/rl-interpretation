from ray import tune

from toolbox import initialize_ray
from toolbox.dece.dece import DECETrainer
from toolbox.marl import MultiAgentEnvWrapper


def _validate_base(
        extra_config,
        test_mode,
        env_name,
        trainer,
        stop=50000,
        local_mode=True,
        name="DELETEME_TEST",
        num_gpus=0
):
    initialize_ray(test_mode=test_mode, local_mode=local_mode,
                   num_gpus=num_gpus)
    num_agents = 3
    env_config = {"env_name": env_name, "num_agents": num_agents}
    config = {
        "seed": 0,
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
    }
    if extra_config:
        config.update(extra_config)
    return tune.run(
        trainer,
        name=name,
        stop={"info/num_steps_sampled": stop},
        config=config
    )


def test_dece(local_mode=False):
    _validate_base(
        trainer=DECETrainer,
        local_mode=local_mode,
        test_mode=True,
        extra_config={
            "num_cpus_per_worker": 0.5,
            "num_workers": 1,
            "num_envs_per_worker": 2,

            # new config:
            # "clip_action_prob_kl": 0.0
        },
        # extra_config={"mode": DIVERSITY_ENCOURAGING},
        env_name="Pendulum-v0",
        stop=10000
    )


# def validate_ceppo():
#     _validate_base(
#         # {"mode": tune.grid_search(OPTIONAL_MODES)}, False, "CartPole-v0",
#         {
#             "mode": tune.grid_search(
#                 # [DISABLE, DISABLE_AND_EXPAND, REPLAY_VALUES,
#                 NO_REPLAY_VALUES]
#                 [REPLAY_VALUES]
#             )
#         },
#         True,
#         "CartPole-v0",
#         CEPPOTrainer
#     )


if __name__ == '__main__':
    test_dece(local_mode=True)
