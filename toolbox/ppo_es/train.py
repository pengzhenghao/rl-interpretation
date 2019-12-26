import argparse

from toolbox.cooperative_exploration.train import train
from toolbox.ppo_es.ppo_es import PPOESTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--env", type=str, default="BipedalWalker-v2")
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--stop", type=float, default=5e6)
    parser.add_argument("--address", type=str, default="")
    args = parser.parse_args()

    if not args.test:
        assert args.exp_name

    ppo_es_config = {
        "num_sgd_iter": 10,
        "num_envs_per_worker": 16,
        "gamma": 0.99,
        "entropy_coeff": 0.001,
        "lambda": 0.95,
        "lr": 2.5e-4,
        "num_gpus": 0.2,
        "num_cpus_per_worker": 0.5,
        "num_cpus_for_driver": 0.8,
        "clip_action_prob_kl": 1
    }

    train(
        extra_config=ppo_es_config,
        trainer=PPOESTrainer,
        env_name=args.env,
        stop=int(args.stop),
        exp_name="DELETEME-TEST" if args.test else args.exp_name,
        num_agents=args.num_agents,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        address=args.address if args.address else None
    )
