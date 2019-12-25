import copy
import json
import logging
import os.path as osp
from collections import OrderedDict

import ray
from ray import tune

from toolbox import initialize_ray
from toolbox.ipd.tnb_rllib import TNBTrainer
from toolbox.process_data import get_latest_checkpoint

logger = logging.getLogger(__file__)

COMMON_CONFIG = {
    "env": "BipedalWalker-v2",
    "novelty_threshold": 0.5,

    "num_sgd_iter": 10,
    "num_envs_per_worker": 16,
    "gamma": 0.99,
    "entropy_coeff": 0.001,
    "lambda": 0.95,
    "lr": 2.5e-4,
    "num_gpus": 0.2,
    "num_cpus_per_worker": 0.5,
    "num_cpus_for_driver": 0.8
}


def train_one_agent(local_dir, agent_name, config, stop, test_mode=False):
    assert ray.is_initialized()
    analysis = tune.run(
        TNBTrainer,
        name=agent_name,
        local_dir=local_dir,
        verbose=2 if test_mode else 1,
        checkpoint_freq=0,
        checkpoint_at_end=True,
        stop=stop,
        config=config,
        reuse_actors=True
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
            ))

    # checkpoint_dict = {'path': PATH, 'iter': The # of iteration}
    checkpoint_dict = get_latest_checkpoint(trial_dir)
    checkpoint = checkpoint_dict['path']
    info = {"checkpoint_dict": checkpoint_dict}
    return analysis, checkpoint, info


def train_one_iteration(iteration_id, exp_name, max_num_agents,
                        parse_agent_result, test_mode=False):
    """Conduct one iteration of evolution. Maximum generated agents is defined
     by max_num_agents"""
    local_dir = osp.join(osp.expanduser("~/ray_results"), exp_name)
    result_dict = OrderedDict()
    checkpoint_dict = OrderedDict()
    agent_info_dict = OrderedDict()
    iteration_info = OrderedDict()
    common_config = copy.deepcopy(COMMON_CONFIG)

    if test_mode:
        common_config['num_gpus'] = 0
        common_config['sample_batch_size'] = 32
        common_config['train_batch_size'] = 128
        common_config['num_sgd_iter'] = 2

    best_reward = float("-inf")

    for agent_id in range(max_num_agents):
        # prepare config
        agent_name = "iter{}_agent{}".format(iteration_id, agent_id)
        agent_config = copy.deepcopy(common_config)
        agent_config["checkpoint_dict"] = json.dumps(checkpoint_dict)
        agent_stop_criterion = dict(
            timesteps_total=int(2e6) if not test_mode else 10000
        )

        # train one agent
        analysis, checkpoint, info = train_one_agent(
            local_dir=local_dir,
            agent_name=agent_name,
            config=agent_config,
            stop=agent_stop_criterion,
            test_mode=test_mode
        )

        prefix = "[Iteration {}, Agent {}/{}](start from 1) " \
                 "Agent <{}>:".format(iteration_id + 1, agent_id + 1,
                                      max_num_agents, agent_name)

        print("{} Finished training.".format(prefix))

        # stop this evolution iteration
        stop_flag, result = parse_agent_result(analysis, prefix)

        # checkpoint would look like: /home/xxx/ray_results/exp_name/
        # iter0_agent0/PPO.../checkpoint-10/checkpoint_10
        checkpoint_dict[agent_name] = checkpoint
        result_dict[agent_name] = result
        agent_info_dict[agent_name] = info
        best_reward = result['current_reward']

        if stop_flag:
            break

    iteration_info['best_reward'] = best_reward
    return result_dict, checkpoint_dict, agent_info_dict, iteration_info


def parse_agent_result_builder(analysis, prefix, prev_reward):
    reward = analysis.dataframe()['episode_reward_mean']
    assert len(reward) == 1
    reward = reward[0]
    ret = reward > prev_reward
    agent_result = {
        "analysis": analysis,
        "current_reward": reward,
        "previous_reward": prev_reward
    }

    print("{} Previous episode_reward_mean is {:.4f}, we have {:.4f} now. "
          "So we choose to {} this agents.".format(
        prefix, prev_reward, reward, "stop" if ret else "continue"))

    return ret, agent_result


if __name__ == '__main__':
    initialize_ray(test_mode=True, local_mode=False)

    prev_reward = float('-inf')

    for iteration_id in range(3):
        def parse_agent_result(analysis, prefix):
            return parse_agent_result_builder(analysis, prefix, prev_reward)


        _, _, _, iteration_info = train_one_iteration(
            iteration_id=iteration_id,
            exp_name="DELETEME_TEST",
            max_num_agents=3,
            parse_agent_result=parse_agent_result,
            test_mode=True
        )
        print("Finished iteration {}! Current best reward {:.4f}, "
              "previous best reward {:.4f}".format(
            iteration_id, iteration_info['best_reward'], prev_reward))

        prev_reward = iteration_info['best_reward']
