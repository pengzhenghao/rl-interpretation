import argparse

from ray import tune

from toolbox.env import get_env_maker
from toolbox.marl import MultiAgentEnvWrapper, on_train_result
from toolbox.marl.adaptive_extra_loss import AdaptiveExtraLossPPOTrainer
from toolbox.marl.extra_loss_ppo_trainer import ExtraLossPPOTrainer
from toolbox.marl.smart_adaptive_extra_loss import \
    SmartAdaptiveExtraLossPPOTrainer
from toolbox.marl.task_novelty_bisector import TNBPPOTrainer
from toolbox.utils import get_local_dir, initialize_ray

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--env", type=str, default="BipedalWalker-v2")
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--num-seeds", type=int, default=0)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    exp_name = args.exp_name
    env_name = args.env
    run_name = args.run

    run_dict = {
        "individual": "PPO",
        "extra_loss": ExtraLossPPOTrainer,
        "adaptive_extra_loss": AdaptiveExtraLossPPOTrainer,
        "smart_adaptive_extra_loss": SmartAdaptiveExtraLossPPOTrainer,
        "off_policy_tnb": TNBPPOTrainer,
        "off_policy_tnb_min_novelty": TNBPPOTrainer,
        "on_policy_tnb": TNBPPOTrainer,
        "on_policy_tnb_min_novelty": TNBPPOTrainer,
        "tnb_4in1": TNBPPOTrainer
    }

    run_specify_config = {
        "individual": {},
        "adaptive_extra_loss": {},
        "smart_adaptive_extra_loss": {
            "num_gpus": 0.2,
            "novelty_loss_param_step": tune.grid_search([0.01, 0.05, 0.1])
        },
        "extra_loss": {
            "novelty_loss_param": tune.grid_search(
                [0.0005, 0.001, 0.005, 0.01, 0.05])
        },
        "off_policy_tnb": {},
        "off_policy_tnb_min_novelty": {
            "novelty_mode": "min"
        },
        "on_policy_tnb": {
            "use_joint_dataset": False
        },
        "on_policy_tnb_min_novelty": {
            "use_joint_dataset": False,
            "novelty_mode": "min"
        },
        "tnb_4in1": {
            "use_joint_dataset": tune.grid_search([True, False]),
            "novelty_mode": tune.grid_search(["min", "mean"])
        }
    }

    run_specify_stop = {
        "individual": {
            "timesteps_total": int(1e7)
        },
        "extra_loss": {
            "timesteps_total": int(1e7)
        },
        "off_policy_tnb": {
            "timesteps_total": int(1e7),
            "episode_reward_mean": 310 * args.num_agents
        }
    }
    run_specify_stop["off_policy_tnb_min_novelty"
    ] = run_specify_stop["off_policy_tnb"]
    run_specify_stop["on_policy_tnb_min_novelty"
    ] = run_specify_stop["off_policy_tnb"]
    run_specify_stop["on_policy_tnb"] = run_specify_stop["off_policy_tnb"]
    run_specify_stop["tnb_4in1"] = run_specify_stop["off_policy_tnb"]
    run_specify_stop["adaptive_extra_loss"] = run_specify_stop["extra_loss"]
    run_specify_stop["smart_adaptive_extra_loss"] = run_specify_stop[
        "extra_loss"]

    assert run_name in run_dict, "--run argument should be in {}, " \
                                 "but you provide {}." \
                                 "".format(run_dict.keys(), run_name)

    initialize_ray(
        num_gpus=args.num_gpus,
        test_mode=args.test_mode,
        object_store_memory=40 * 1024 * 1024 * 1024,
        # temp_dir="/data1/pengzh/tmp"
    )

    policy_names = ["ppo_agent{}".format(i) for i in range(args.num_agents)]

    tmp_env = get_env_maker(env_name)()
    default_policy = (
        None, tmp_env.observation_space, tmp_env.action_space, {}
    )

    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": env_name,
            "agent_ids": policy_names
        },
        "log_level": "DEBUG" if args.test_mode else "ERROR",
        # "num_gpus": 0.45,
        "num_gpus": 1,
        "num_cpus_per_worker": 2,
        "num_cpus_for_driver": 1,
        "num_envs_per_worker": 16,
        "sample_batch_size": 256,
        "multiagent": {
            "policies": {i: default_policy
                         for i in policy_names},
            "policy_mapping_fn": lambda aid: aid,
        },
        "callbacks": {
            "on_train_result": on_train_result
        },
        "num_sgd_iter": 10,
        "seed": tune.grid_search(list(range(args.num_seeds)))
        if args.num_seeds != 0 else 0
    }
    config.update(run_specify_config[run_name])

    tune.run(
        run_dict[run_name],
        local_dir=get_local_dir(),
        name=exp_name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        stop=run_specify_stop[run_name],
        config=config,
    )
