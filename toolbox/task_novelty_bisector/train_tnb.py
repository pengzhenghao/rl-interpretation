import pickle

import ray

from toolbox import initialize_ray
from toolbox.ipd.train_tnb import TNBTrainer, parse_agent_result_builder, \
    train_one_iteration


def main(
        exp_name,
        max_num_agents,
        timesteps_total,
        common_config,
        ray_init,
        test_mode=False,
        disable_early_stop=False,
        trainer=TNBTrainer
):
    prev_reward = float('-inf')
    preoccupied_checkpoints = {}
    info_dict = {}

    def parse_agent_result(analysis, prefix):
        return parse_agent_result_builder(analysis, prefix, prev_reward)

    # train one iteration (at most max_num_agents will be trained)
    result_dict, checkpoint_dict, _, iteration_info = \
        train_one_iteration(
            trainer=trainer,
            iteration_id=0,
            exp_name=exp_name if not test_mode else "DELETEME-TEST",
            max_num_agents=max_num_agents if not test_mode else 3,
            parse_agent_result=parse_agent_result,
            timesteps_total=timesteps_total,
            common_config=common_config,
            ray_init=ray_init,
            preoccupied_checkpoints=preoccupied_checkpoints,
            test_mode=test_mode,
            disable_early_stop=disable_early_stop
        )

    print("Finish! Current best reward {:.4f}, best agent {}".format(
        iteration_info['best_reward'], iteration_info['best_agent']))

    # Save data
    for agent_name, result in result_dict.items():
        info_dict[agent_name] = dict(
            dataframe=next(iter(result['analysis'].trial_dataframes.values())),
            reward=result['current_reward'],
            checkpoint=checkpoint_dict[agent_name]
        )
    agent_dict_file_name = "{}_agent_dict.pkl".format(exp_name)
    with open(agent_dict_file_name, 'wb') as f:
        pickle.dump(info_dict, f)
    print("Data has been saved at: {}. "
          "Terminate the program.".format(agent_dict_file_name))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default="")
    # parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--timesteps", type=float, default=1e6)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--max-num-agents", type=int, default=5)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v2")

    # parser.add_argument("--address", type=str, default="")

    # You may need to grid search
    # parser.add_argument("--novelty-threshold", type=float, default=0.5)
    # parser.add_argument("--use-preoccupied-agent", action="store_true")
    # parser.add_argument("--disable-tnb", action="store_true")
    # parser.add_argument("--use-tnb-plus", action="store_true")
    # parser.add_argument("--max-not-improve-iterations", type=int, default=3)

    args = parser.parse_args()

    if not args.test_mode:
        assert args.exp_name


    def ray_init():
        ray.shutdown()
        initialize_ray(
            test_mode=args.test_mode,
            local_mode=False,
            num_gpus=args.num_gpus if not args.address else None,
            redis_address=args.address if args.address else None
        )


    walker_config = {
        "use_preoccupied_agent": args.use_preoccupied_agent,
        "disable_tnb": args.disable_tnb,
        "env": args.env_name,
        "use_tnb_plus": args.use_tnb_plus,
        "novelty_threshold": args.novelty_threshold,
        # "novelty_threshold": 1.1,
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": 0.0001,
        'sample_batch_size': 256,
        'sgd_minibatch_size': 4096,
        'train_batch_size': 65536,
        "num_gpus": 0.4,
        "num_cpus_per_worker": 0.4,
        "num_cpus_for_driver": 0.4,
        "num_envs_per_worker": 16,
        'num_workers': 8,
    }

    if args.env_name == "Walker2d-v3":
        common_config = walker_config
    else:
        raise NotImplementedError()

    main(
        exp_name=args.exp_name,
        num_iterations=args.num_iterations,
        max_num_agents=args.max_num_agents,
        timesteps_total=int(args.timesteps),
        common_config=common_config,
        max_not_improve_iterations=args.max_not_improve_iterations,
        ray_init=ray_init,
        test_mode=args.test_mode
    )
