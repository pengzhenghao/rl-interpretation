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
        test_mode=False,
        address=None,
        **kwargs
):
    # assert isinstance(stop, int)
    if address is not None:
        num_gpus = None
    initialize_ray(
        test_mode=test_mode,
        local_mode=False,
        num_gpus=num_gpus,
        address=address
    )
    env_config = {"env_name": env_name, "num_agents": num_agents}
    config = {
        "seed": tune.grid_search([i * 100 for i in range(num_seeds)]),
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "log_level": "DEBUG" if test_mode else "INFO"
    }
    if extra_config:
        config.update(extra_config)

    analysis = tune.run(
        trainer,
        local_dir=get_local_dir(),
        name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=10,
        checkpoint_score_attr="episode_reward_mean",
        checkpoint_at_end=True,
        stop={"info/num_steps_sampled": stop}
        if isinstance(stop, int) else stop,
        config=config,
        max_failures=20,
        reuse_actors=False,
        **kwargs
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
    parser.add_argument("--stop", type=float, default=5e6)
    parser.add_argument("--address", type=str, default="")
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
        "num_gpus": 0.2,
        "num_cpus_per_worker": 0.5,
        "num_cpus_for_driver": 0.8,
        "clip_action_prob_kl": 1
    }

    clip_action_prob_kl = None
    if args.mode == "all":
        mode = tune.grid_search(OPTIONAL_MODES)
        num_agents = args.num_agents
    # elif args.mode == "change_num_agents":
    #     mode = tune.grid_search([REPLAY_VALUES, NO_REPLAY_VALUES])
    #     num_agents = tune.grid_search(list(range(2, args.num_agents + 1)))
    # elif args.mode == "change_num_agents_disable_and_expand":
    #     mode = DISABLE_AND_EXPAND
    #     num_agents = tune.grid_search(list(range(2, args.num_agents + 1)))
    #     num_gpus = 0.5
    elif args.mode == "three":
        mode = tune.grid_search([REPLAY_VALUES, NO_REPLAY_VALUES, DISABLE])
        clip_action_prob_kl = tune.grid_search([0.01, 0.1, 1])
        num_agents = tune.grid_search([3, 5, 7])
    elif args.mode == "search_replay_values":
        mode = REPLAY_VALUES
        clip_action_prob_kl = tune.grid_search([0.1, 1, 10])
        ceppo_config["clip_action_prob_ratio"] = tune.grid_search([0.5, 1, 2])
        num_agents = tune.grid_search([3, 5])
    elif args.mode == "new_adv_1221":
        mode = REPLAY_VALUES
        clip_action_prob_kl = tune.grid_search([0, 0.1, 100])
        # ceppo_config["clip_action_prob_ratio"] = tune.grid_search([0.5, 1,
        # 2, 10])
        num_agents = tune.grid_search([3])
        ceppo_config["grad_clip"] = tune.grid_search([0.5, 1, 10])
    elif args.mode == "clip_advantage":
        mode = REPLAY_VALUES
        clip_action_prob_kl = tune.grid_search([0, 0.1, 100])
        # ceppo_config["clip_action_prob_ratio"] = tune.grid_search([0.5, 1,
        # 2, 10])
        num_agents = tune.grid_search([3])
        ceppo_config["grad_clip"] = tune.grid_search([0.5, 1, 10])
        ceppo_config["clip_advantage"] = True
    elif args.mode == "baseline_shrink":
        mode = DISABLE_AND_EXPAND
        clip_action_prob_kl = tune.grid_search([0.01, 0.1, 1])
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
        stop=int(args.stop),
        exp_name="DELETEME-TEST" if args.test else args.exp_name,
        num_agents=num_agents,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        address=args.address if args.address else None
    )
