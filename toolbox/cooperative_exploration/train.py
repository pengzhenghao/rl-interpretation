import argparse
import pickle

from ray import tune

from toolbox import initialize_ray, get_local_dir
from toolbox.cooperative_exploration.ceppo import OPTIONAL_MODES, CEPPOTrainer
from toolbox.marl import MultiAgentEnvWrapper


def train(
        extra_config,
        trainer,
        env_name,
        stop,
        exp_name,
        num_agents,
        num_seeds,
        num_gpus,
        test_mode=False
):
    assert isinstance(stop, int)
    initialize_ray(test_mode=test_mode, local_mode=False, num_gpus=num_gpus)

    policy_names = ["agent{}".format(i) for i in range(num_agents)]
    env_config = {"env_name": env_name, "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "seed": tune.grid_search([i * 100 for i in range(num_seeds)]),
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

    analyasis = tune.run(
        trainer,
        local_dir=get_local_dir(),
        name=exp_name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        stop={"info/num_steps_sampled": stop},
        config=config,
        reuse_actors=True
    )

    path = "{}-{}-{}ts-{}agents.pkl".format(
        exp_name, env_name, stop, num_agents
    )
    with open(path, "wb") as f:
        data = analyasis.fetch_trial_dataframes()
        pickle.dump(data, f)
        print("Result is saved at: <{}>".format(path))

    return analyasis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--env", type=str, default="BipedalWalker-v2")
    parser.add_argument("--exp-name", type=str, required=True)
    args = parser.parse_args()

    ceppo_config = {
        "num_sgd_iter": 10,
        "num_envs_per_worker": 16,
        "gamma": 0.99,
        "entropy_coeff": 0.001,
        "lambda": 0.95,
        "lr": 2.5e-4,
        "mode": tune.grid_search(OPTIONAL_MODES),
        "num_gpus": 0.24,
    }

    train(
        extra_config=ceppo_config,
        trainer=CEPPOTrainer,
        env_name=args.env,
        stop=int(5e6) if not args.test else 1000,
        exp_name="DELETEME-TEST" if args.test else args.exp_name,
        num_agents=args.num_agents,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test
    )
