import argparse
import pickle

from ray import tune

from toolbox import initialize_ray, get_local_dir
from toolbox.cooperative_exploration.ceppo import *
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
    env_config = {"env_name": env_name, "num_agents": num_agents}
    config = {
        "seed": tune.grid_search([i * 100 for i in range(num_seeds)]),
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
    }
    if extra_config:
        config.update(extra_config)

    analysis = tune.run(
        trainer,
        local_dir=get_local_dir(),
        name=exp_name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        stop={"info/num_steps_sampled": stop},
        config=config,
        max_failures=20,
        reuse_actors=True
    )

    path = "{}-{}-{}ts-{}agents.pkl".format(
        exp_name, env_name, stop, num_agents
    )
    with open(path, "wb") as f:
        data = analysis.fetch_trial_dataframes()
        pickle.dump(data, f)
        print("Result is saved at: <{}>".format(path))

    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--env", type=str, default="BipedalWalker-v2")
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--mode", type=str, default="all")
    args = parser.parse_args()

    if not args.test:
        assert args.exp_name

    ceppo_config = {
        "num_sgd_iter": 10,
        "num_envs_per_worker": 16,
        "gamma": 0.99,
        "entropy_coeff": 0.001,
        "lambda": 0.95,
        "lr": 2.5e-4,
        "num_gpus": 0.15,
        "num_cpus_per_worker": 0.5,
        # "num_cpus_for_driver": 0.8,
        "clip_action_prob_kl": 1
    }

    clip_action_prob_kl = None
    if args.mode == "all":
        mode = tune.grid_search(OPTIONAL_MODES)
        num_agents = args.num_agents
    elif args.mode == "change_num_agents":
        mode = tune.grid_search([REPLAY_VALUES, NO_REPLAY_VALUES])
        num_agents = tune.grid_search(list(range(2, args.num_agents + 1)))
    elif args.mode == "change_num_agents_disable_and_expand":
        mode = DISABLE_AND_EXPAND
        num_agents = tune.grid_search(list(range(2, args.num_agents + 1)))
        num_gpus = 0.5
    elif args.mode == "replay_values":
        mode = REPLAY_VALUES
        clip_action_prob_kl = tune.grid_search([0.01, 0.1, 1])
        num_agents = tune.grid_search([3, 5, 7])
    elif args.mode == "baseline":
        mode = tune.grid_search([DISABLE, NO_REPLAY_VALUES])
        num_agents = tune.grid_search([3, 5, 7])
    else:
        raise NotImplementedError()

    # ceppo_config["num_agents"] = num_agents
    ceppo_config["mode"] = mode
    if clip_action_prob_kl:
        ceppo_config["clip_action_prob_kl"] = clip_action_prob_kl

    train(
        extra_config=ceppo_config,
        trainer=CEPPOTrainer,
        env_name=args.env,
        stop=int(5e6),
        exp_name="DELETEME-TEST" if args.test else args.exp_name,
        num_agents=num_agents,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test
    )
