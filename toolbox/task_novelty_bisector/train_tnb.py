import copy
import json
import logging
import os
import os.path as osp
import pickle
from collections import OrderedDict

import numpy as np
import ray
from ray import tune

from toolbox import initialize_ray, get_train_parser
from toolbox.ipd.train_tnb import TNBTrainer
from toolbox.process_data import get_latest_checkpoint

logger = logging.getLogger(__name__)


def train_one_agent(
        trainer, local_dir, agent_name, config, stop, test_mode=False
):
    assert ray.is_initialized()
    analysis = tune.run(
        trainer,
        name=agent_name,
        local_dir=local_dir,
        verbose=1,
        checkpoint_freq=0,
        checkpoint_at_end=True,
        stop=stop,
        config=config
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
        trainer,
        iteration_id,
        exp_name,
        max_num_agents,
        parse_agent_result,
        timesteps_total,
        common_config,
        ray_init,
        preoccupied_checkpoints=None,
        test_mode=False,
        disable_early_stop=False
):
    """Conduct one iteration of evolution. Maximum number of generated agents
     is defined by max_num_agents"""
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
        common_config['sgd_minibatch_size'] = 32
        common_config['train_batch_size'] = 128
        common_config['num_sgd_iter'] = 2

    best_reward = float("-inf")

    for agent_id in range(max_num_agents):
        ray_init()

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
            trainer=trainer,
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

        checkpoint_dict[agent_name] = {
            "path": checkpoint,
            "reward": result['current_reward']
        }  # this dict is used at AgentPoolMixin._lazy_initialize

        result_dict[agent_name] = result
        agent_info_dict[agent_name] = info

        if best_reward < result['current_reward']:
            iteration_info['best_agent'] = agent_name
            best_reward = result['current_reward']

        if stop_flag and disable_early_stop:
            print(
                "{} Though we choose to STOP here, but we set"
                " disable_eary_stop in order to align different "
                "expeirments, therefore we CONTINUE this iteration."
                "".format(prefix)
            )

        if (stop_flag) and (not disable_early_stop):
            break

    iteration_info['best_reward'] = best_reward
    return result_dict, checkpoint_dict, agent_info_dict, iteration_info


def parse_agent_result_builder(analysis, prefix, prev_reward):
    reward = analysis.dataframe()['episode_reward_mean']
    assert len(reward) == 1, "This workflow only run single agent. " \
                             "But you have: {}".format(reward)
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
    parser = get_train_parser()
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--timesteps", type=float, default=1e6)
    parser.add_argument("--max-num-agents", type=int, default=5)
    args = parser.parse_args()

    env_name = "{}-{}".format(args.env_name, args.env_name)


    def ray_init():
        ray.shutdown()
        initialize_ray(test_mode=args.test_mode, num_gpus=args.num_gpus)


    large = env_name in ["Walker2d-v3", "Hopper-v3"]
    if large:
        stop = int(2e7)
    elif env_name == "Humanoid-v3":
        stop = int(2e7)
    else:
        stop = int(5e6)

    config = {
        "env": args.env_name,
        "kl_coeff": 1.0,
        "num_sgd_iter": 10,
        "lr": 0.0001,
        'sample_batch_size': 200 if large else 50,
        'sgd_minibatch_size': 100 if large else 64,
        'train_batch_size': 10000 if large else 2048,
        "num_gpus": 1,
        "num_cpus_per_worker": 2,
        "num_cpus_for_driver": 1 if large else 2,
        "num_envs_per_worker": 8 if large else 5,
        'num_workers': 8 if large else 1,
        "use_tnb_plus": False
    }

    main(
        exp_name=args.exp_name,
        num_iterations=args.num_iterations,
        max_num_agents=args.max_num_agents,
        timesteps_total=int(args.timesteps),
        common_config=config,
        max_not_improve_iterations=args.max_not_improve_iterations,
        ray_init=ray_init,
        test_mode=args.test_mode
    )
