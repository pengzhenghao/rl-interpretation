import argparse
import copy
import os
import os.path as osp

from ray import tune

from toolbox.ipd.interior_policy_differentiation import IPDEnv, on_episode_end
from toolbox.utils import get_local_dir, initialize_ray

parser = argparse.ArgumentParser()
parser.add_argument("--yaml-path", type=str, required=True)
parser.add_argument("--exp-name", type=str, required=True)
parser.add_argument("--env", type=str, default="BipedalWalker-v2")
parser.add_argument("--num-seeds", type=int, default=3)
parser.add_argument("--num-gpus", type=int, default=4)
parser.add_argument("--test-mode", action="store_true")
args = parser.parse_args()

ipd_common_config = {
    "env": IPDEnv,
    "env_config": {
        "env_name": "BipedalWalker-v2",
        "novelty_threshold": None,
        "yaml_path": args.yaml_path
    },
    "callbacks": {"on_episode_end": on_episode_end}
}

algo_config = {"BipedalWalker-v2": {
    "num_sgd_iter": 10,
    "num_envs_per_worker": 16,
    "gamma": 0.99,
    "entropy_coeff": 0.001,
    "lambda": 0.95,
    "lr": 2.5e-4,
}}

stop_config = {"BipedalWalker-v2": {
    "timesteps_total": int(1e7)
}}

env_name = args.env

config = algo_config[env_name]
config.update(ipd_common_config)

# FIXME add codes here to update novelty_threshold at config.env_config
novelty_threshold = None
config['env_config']['novelty_threshold'] = novelty_threshold


def _search_ckpt(save_path, input_exp_name, iter_id, seed):
    """

    :param save_path: /home/xxx/ray_results/ipd1111/
    :param input_exp_name: ipd1111_seed0_iter3
    :param iter_id: 1
    :param seed: 0
    :return: /home/xxx/ray_results/ipd1111/ipd1111_seed0_iter3/PPO_xxxxx/checkpoint-100/checkpoint_100
    """

    # look for new restore_checkpoint.
    # FIXME assert no iter=0 checkpoint can be found otherwise should
    #  have ckpt

    assert osp.exists(osp.dirname(save_path))

    sub_exp_path = osp.join(save_path, input_exp_name) # ../exp/exp_seed0_iter3
    if (not osp.exists(save_path)) or \
            (not osp.exists(sub_exp_path)) or \
            (not os.listdir(sub_exp_path)):
        return None

    # should be ['ppo_xxx', .., 'experiment_..']
    dirs = [d for d in os.listdir(sub_exp_path) if osp.isdir(d)]

    if len(dirs) != 1:
        print("We expecte there is only one trial")
        return None

    trail_dir = dirs[0]




def train_one_iteration(iter_id, exp_name, num_gpus, test_mode):
    assert isinstance(iter_id, int)
    local_dir = get_local_dir() if get_local_dir() else "~/ray_results"
    local_dir = os.path.expanduser(local_dir)
    save_path = os.path.join(local_dir, exp_name)

    for i in range(args.num_seeds):
        input_exp_name = exp_name + "_seed{}_iter{}".format(i, iter_id)
        restore = _search_ckpt(save_path, input_exp_name, iter_id, i)
        assert (restore is None) or (osp.exists(restore))

        if restore is not None:
            print("Load checkpoint: <{}>. Now start at append it to "
                  "the agent pool.")

        tmp_config = copy.deepcopy(config)
        tmp_config.update(seed=i)
        initialize_ray(num_gpus=num_gpus, test_mode=test_mode)
        tune.run(
            name=input_exp_name,
            verbose=1,
            local_dir=save_path,
            checkpoint_freq=10,
            checkpoint_at_end=True,
            stop=stop_config[env_name],
            config=config
        )


def test_train_ipd():


    for iter_id in range(10):
        train_one_iteration()


print("Finish!")
