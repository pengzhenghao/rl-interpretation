import argparse

from ray import tune

from toolbox.cooperative_exploration.train import train
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.ppo_es.ppo_es import PPOESTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--env", type=str, default="Humanoid-v2")
    parser.add_argument("--exp-name", type=str, default="")
    # parser.add_argument("--mode", type=str, default="all")
    # parser.add_argument("--stop", type=float, default=5e6)
    parser.add_argument("--address", type=str, default="")
    args = parser.parse_args()

    if not args.test:
        assert args.exp_name

    humanoid_config = {
        # can change
        "update_steps": tune.grid_search([100000, 200000, 500000]),
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": args.env,
            "num_agents": args.num_agents
        },

        # should be fixed
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": 0.0001,
        "gamma": 0.995,
        "lambda": 0.95,
        'sgd_minibatch_size': 4096,
        'train_batch_size': 65536,
        "sample_batch_size": 256,
        "num_envs_per_worker": 16,
        "num_cpus_per_worker": 0.5,
        'num_workers': 12,
        "num_gpus": 0.8
    }

    train(
        extra_config=humanoid_config,
        trainer=PPOESTrainer,
        env_name=args.env,

        stop={
            "episode_reward_mean": 6000,
            "timesteps_total": int(2e8)
        },
        exp_name="DELETEME-TEST" if args.test else args.exp_name,
        num_agents=args.num_agents if not args.test else 3,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        address=args.address if args.address else None
    )
