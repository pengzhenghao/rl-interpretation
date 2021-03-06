import logging
import pickle

import ray

from toolbox import initialize_ray
from toolbox.task_novelty_bisector import TNBTrainer, train_one_iteration, \
    parse_agent_result_builder

"""
TNB-ES training basic workflow:

    1. The outer loops is the evolution iteration. In each iteration we will
    generate a population of agents, which leverages the diversity-seeking
    algorithms to make them as diverse as possible.

    2. We will take the best agent within a population (that is, within an
    iteration) to make it as a 'seed' for next evolution iteration.

    3. How to use the best agent as a seed? We use it as a preoccupied 
    comparing subject in the new iteration. 

    For example, in the first iteration, there do not exist such 'comparing 
    subject', so the first agent is trained from sketch without any 
    diversity-seeking incentive. But in the second evolution iteration, the
    first agent CAN compare itself with the 'preoccupied comparing subject', 
    that is the best agent in the previous iteration. By this way, we can 
    use the best agent in each iteration as the 'seed' for next iteration.
"""

logger = logging.getLogger(__file__)


def main(
        exp_name,
        num_iterations,
        max_num_agents,
        max_not_improve_iterations,
        timesteps_total,
        common_config,
        ray_init,
        test_mode=False,
        disable_early_stop=False,
        trainer=TNBTrainer
):
    prev_reward = float('-inf')
    prev_agent = None
    preoccupied_checkpoints = {}
    not_improve_counter = 0
    info_dict = {}
    iteration_result = []

    for iteration_id in range(num_iterations):
        print(
            "[Iteration {}/{}] Start! Previous best reward {:.4f},"
            " best agent {}. Current preoccupied_checkpoints keys {}.".format(
                iteration_id + 1, num_iterations, prev_reward, prev_agent,
                preoccupied_checkpoints.keys()
            )
        )

        def parse_agent_result(analysis, prefix):
            return parse_agent_result_builder(analysis, prefix, prev_reward)

        # clear up existing worker at previous iteration.
        # ray_init()

        # train one iteration (at most max_num_agents will be trained)
        result_dict, checkpoint_dict, _, iteration_info = \
            train_one_iteration(
                trainer=trainer,
                iteration_id=iteration_id,
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

        # save necessary data
        for agent_name, result in result_dict.items():
            info = dict(
                dataframe=next(
                    iter(result['analysis'].trial_dataframes.values())
                ),
                reward=result['current_reward'],
                checkpoint=checkpoint_dict[agent_name]
            )
            info_dict[agent_name] = info
        iteration_result.append(iteration_info)

        # get the best agent
        current_reward = iteration_info['best_reward']
        best_agent = iteration_info['best_agent']
        best_agent_checkpoint = checkpoint_dict[best_agent]

        print(
            "[Iteration {}/{}] Finish! Current best reward {:.4f},"
            " best agent {}, previous best reward {:.4f}, "
            "previous best agent {}. Not improve performance for {} "
            "iterations.".format(
                iteration_id + 1, num_iterations, current_reward, best_agent,
                prev_reward, prev_agent, not_improve_counter + int(
                    (current_reward <= prev_reward) and
                    (prev_agent is not None)
                )
            )
        )

        # early stop mechanism in iteration-level
        if (current_reward > prev_reward) or (prev_agent is None):
            prev_reward = current_reward
            prev_agent = best_agent
            preoccupied_checkpoints = {best_agent: best_agent_checkpoint}
            not_improve_counter = 0
        else:
            not_improve_counter += 1

        if not_improve_counter >= max_not_improve_iterations:
            print(
                "[Iteration {}/{}] Stop Iterating! Current best reward {:.4f},"
                " best agent {}, previous best reward {:.4f}, "
                "previous best agent {}. Not improve performance for {} "
                "iterations. Exceed the maximum number of not-improving "
                "iteration {}, so we stop the whole program.".format(
                    iteration_id + 1, num_iterations, current_reward,
                    best_agent, prev_reward, prev_agent,
                    not_improve_counter + int(
                        (current_reward <= prev_reward) and
                        (prev_agent is not None)
                    ), max_not_improve_iterations
                )
            )
            break

    agent_dict_file_name = "{}_agent_dict.pkl".format(exp_name)
    with open(agent_dict_file_name, 'wb') as f:
        pickle.dump(info_dict, f)

    iteration_result_file_name = "{}_iteration_result.pkl".format(exp_name)
    with open(iteration_result_file_name, 'wb') as f:
        pickle.dump(info_dict, f)

    print(
        "Finish {} iterations! Data has been saved at: {}. "
        "Terminate the program.".format(
            num_iterations, (agent_dict_file_name, iteration_result_file_name)
        )
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default="")
    # parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--timesteps", type=float, default=1e6)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--max-num-agents", type=int, default=10)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v2")

    parser.add_argument("--address", type=str, default="")

    # You may need to grid search
    parser.add_argument("--novelty-threshold", type=float, default=0.5)
    parser.add_argument("--use-preoccupied-agent", action="store_true")
    parser.add_argument("--disable-tnb", action="store_true")
    parser.add_argument("--use-tnb-plus", action="store_true")
    parser.add_argument("--max-not-improve-iterations", type=int, default=3)

    args = parser.parse_args()

    if not args.test_mode:
        assert args.exp_name

    ppo_config = {
        "use_preoccupied_agent": args.use_preoccupied_agent,
        "env": args.env_name,
        "disable_tnb": args.disable_tnb,
        "use_tnb_plus": args.use_tnb_plus,
        "novelty_threshold": args.novelty_threshold,
        "num_sgd_iter": 10,
        "num_envs_per_worker": 16,
        "gamma": 0.99,
        "entropy_coeff": 0.001,
        "lambda": 0.95,
        "lr": 2.5e-4,
        "num_gpus": 0.3,
        # "novelty_threshold": 0.5
    }

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

    if args.env_name == "BipedalWalker-v2":
        common_config = ppo_config
    elif args.env_name == "Walker2d-v3":
        common_config = walker_config
    else:
        raise NotImplementedError()


    def ray_init():
        ray.shutdown()
        initialize_ray(
            test_mode=args.test_mode,
            local_mode=False,
            num_gpus=args.num_gpus if not args.address else None,
            redis_address=args.address if args.address else None
        )


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
    """TODO: pengzh
    Here a brief sketch that we need to do:
        [V] 1. allow restore the previous-iteration best agent.
        [ ] 2. make sure agents is saved.
        [ ] 3. data parsing
    """
