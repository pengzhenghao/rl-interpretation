import copy
import json
import logging
import os
import os.path as osp
from collections import OrderedDict

import numpy as np
import ray
from ray import tune

from toolbox import initialize_ray
from toolbox.ipd.tnb import TNBTrainer
from toolbox.process_data import get_latest_checkpoint
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


def train_one_agent(local_dir, agent_name, config, stop, test_mode=False):
    assert ray.is_initialized()
    analysis = tune.run(
        TNBTrainer,
        name=agent_name,
        local_dir=local_dir,
        verbose=2,
        checkpoint_freq=0,
        checkpoint_at_end=True,
        stop=stop,
        config=config,
        reuse_actors=True,
    )

    # one experiments may have multiple trials, due to some error that
    # corrupt the previous programs. But it doesn't matter since we take the
    # best out of them.
    trial_dir = analysis.get_best_logdir("episode_reward_mean")
    if len(analysis.trials) != 1:
        logger.warning(
            "We found {} trails at the same dir. The runner data: \n{}\nThe "
            "trails: {}".format(
                len(analysis.trials), analysis.runner_data(),
                analysis.trial_dataframes.keys()
            )
        )

    # checkpoint_dict = {'path': PATH, 'iter': The # of iteration}
    checkpoint_dict = get_latest_checkpoint(trial_dir)
    checkpoint = checkpoint_dict['path']
    info = {"checkpoint_dict": checkpoint_dict, "checkpoint_path": checkpoint}
    return analysis, checkpoint, info


def train_one_iteration(
        iteration_id,
        exp_name,
        max_num_agents,
        parse_agent_result,
        timesteps_total,
        common_config,
        preoccupied_checkpoints=None,
        test_mode=False
):
    """Conduct one iteration of evolution. Maximum generated agents is defined
     by max_num_agents"""
    local_dir = osp.join(osp.expanduser("~/ray_results"), exp_name)
    os.makedirs(local_dir, exist_ok=True)

    result_dict = OrderedDict()
    checkpoint_dict = OrderedDict()
    agent_info_dict = OrderedDict()
    iteration_info = OrderedDict()

    if preoccupied_checkpoints:
        assert isinstance(preoccupied_checkpoints, dict)
        assert len(preoccupied_checkpoints) == 1
        checkpoint_dict.update(preoccupied_checkpoints)

    if test_mode:
        common_config['num_gpus'] = 0
        common_config['sample_batch_size'] = 32
        common_config['train_batch_size'] = 128
        common_config['num_sgd_iter'] = 2

    best_reward = float("-inf")

    for agent_id in range(max_num_agents):
        agent_name = "iter{}_agent{}".format(iteration_id, agent_id)
        prefix = "[Iteration {}, Agent {}/{}](start from 1) " \
                 "Agent <{}>:".format(iteration_id + 1, agent_id + 1,
                                      max_num_agents, agent_name)

        print("{} Start training.".format(prefix))

        # prepare config
        agent_config = copy.deepcopy(common_config)
        agent_config["checkpoint_dict"] = json.dumps(checkpoint_dict)
        agent_stop_criterion = dict(
            timesteps_total=int(timesteps_total) if not test_mode else 10000
        )

        # train one agent
        analysis, checkpoint, info = train_one_agent(
            local_dir=local_dir,
            agent_name=agent_name,
            config=agent_config,
            stop=agent_stop_criterion,
            test_mode=test_mode
        )

        print("{} Finished training.".format(prefix))

        # stop this evolution iteration
        stop_flag, result = parse_agent_result(analysis, prefix)

        # checkpoint would look like: /home/xxx/ray_results/exp_name/
        # iter0_agent0/PPO.../checkpoint-10/checkpoint_10
        assert agent_name not in checkpoint_dict
        checkpoint_dict[agent_name] = checkpoint
        result_dict[agent_name] = result
        agent_info_dict[agent_name] = info

        if best_reward < result['current_reward']:
            iteration_info['best_agent'] = agent_name
            best_reward = result['current_reward']

        if stop_flag:
            break

    iteration_info['best_reward'] = best_reward
    return result_dict, checkpoint_dict, agent_info_dict, iteration_info


def parse_agent_result_builder(analysis, prefix, prev_reward):
    reward = analysis.dataframe()['episode_reward_mean']
    assert len(reward) == 1
    reward = reward[0]
    if not np.isfinite(reward):
        reward = float("-inf")
    ret = reward > prev_reward
    agent_result = {
        "analysis": analysis,
        "current_reward": reward,
        "previous_reward": prev_reward
    }

    print(
        "{} Previous episode_reward_mean is {:.4f}, we have {:.4f} now. "
        "So we choose to {} training.".format(
            prefix, prev_reward, reward, "STOP" if ret else "CONTINUE"
        )
    )

    return ret, agent_result


def main(
        exp_name,
        num_iterations,
        max_num_agents,
        timesteps_total,
        common_config,
        test_mode=False,
        _call_shutdown=False,
):
    prev_reward = float('+inf')
    preoccupied_checkpoints = None

    for iteration_id in range(num_iterations):
        print(
            "Start iteration {}/{}! Previous best reward {:.4f}.".format(
                iteration_id + 1, num_iterations,
                prev_reward
            )
        )

        def parse_agent_result(analysis, prefix):
            return parse_agent_result_builder(analysis, prefix, prev_reward)

        # clear up existing worker at previous iteration.
        # FIXME we don't know whether this would destroy others.
        if _call_shutdown:
            ray.shutdown()
        initialize_ray(
            test_mode=args.test_mode,
            local_mode=False,
            num_gpus=args.num_gpus if not args.address else None,
            redis_address=args.address if args.address else None
        )

        result_dict, checkpoint_dict, agent_info_dict, iteration_info = \
            train_one_iteration(
                iteration_id=iteration_id,
                exp_name=exp_name if not test_mode else "DELETEME-TEST",
                max_num_agents=max_num_agents if not test_mode else 3,
                parse_agent_result=parse_agent_result,
                timesteps_total=timesteps_total,
                common_config=common_config,
                preoccupied_checkpoints=preoccupied_checkpoints,
                test_mode=test_mode
            )

        best_agent = iteration_info['best_agent']
        best_agent_checkpoint = agent_info_dict[best_agent]['checkpoint_path']
        preoccupied_checkpoints = {best_agent: best_agent_checkpoint}

        print(
            "Finished iteration {}/{}! Current best reward {:.4f},"
            " best agent {}, previous best reward {:.4f}.".format(
                iteration_id + 1, num_iterations,
                iteration_info['best_reward'], best_agent,
                prev_reward
            )
        )
        prev_reward = iteration_info['best_reward']

    print("Finish {} iterations! Terminate the program.".format(
        num_iterations))
    ### TODO
    ### TODO
    ### TODO
    ### TODO
    ### TODO
    ### TODO
    ### TODO You need to save the result at some place!!!!!
    ### TODO
    ### TODO
    ### TODO
    ### TODO
    ### TODO
    ### TODO
    ### TODO
    ### TODO
    ### TODO
    ### TODO



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default="")
    # parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=1e6)
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--max-num-agents", type=int, default=10)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--shutdown", action="store_true")

    parser.add_argument("--address", type=str, default="")

    # You may need to grid search
    parser.add_argument("--novelty-threshold", type=float, default=0.5)
    parser.add_argument("--use-preoccupied-agent", action="store_true")

    args = parser.parse_args()

    if not args.test_mode:
        assert args.exp_name

    common_config = {
        "novelty_threshold": args.novelty_threshold,
        "use_preoccupied_agent": args.use_preoccupied_agent,

        # do not change
        "env": "BipedalWalker-v2",
        "num_sgd_iter": 10,
        "num_envs_per_worker": 16,
        "gamma": 0.99,
        "entropy_coeff": 0.001,
        "lambda": 0.95,
        "lr": 2.5e-4,
        "num_gpus": 1,
        "num_cpus_per_worker": 0.5,
        "num_cpus_for_driver": 0.8
    }


    main(
        exp_name=args.exp_name,
        num_iterations=args.num_iterations,
        max_num_agents=args.max_num_agents,
        timesteps_total=args.timesteps,
        common_config=common_config,
        test_mode=args.test_mode,
        _call_shutdown=args.shutdown
    )
    """TODO: pengzh
    Here a brief sketch that we need to do:
        [V] 1. allow restore the previous-iteration best agent.
        [ ] 2. make sure agents is saved.
        [ ] 3. data parsing
    """
