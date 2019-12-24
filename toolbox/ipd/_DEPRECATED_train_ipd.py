import copy
import copy
import os
import os.path as osp

from ray import tune

from toolbox.ipd.interior_policy_differentiation import IPDEnv, on_episode_end
from toolbox.process_data.process_data import read_yaml, \
    get_latest_checkpoint, \
    save_yaml
from toolbox.utils import get_local_dir, initialize_ray


# parser = argparse.ArgumentParser()
# parser.add_argument("--yaml-path", type=str, required=True)
# parser.add_argument("--exp-name", type=str, required=True)
# parser.add_argument("--env", type=str, default="BipedalWalker-v2")
# parser.add_argument("--num-seeds", type=int, default=3)
# parser.add_argument("--num-gpus", type=int, default=4)
# parser.add_argument("--test-mode", action="store_true")
# args = parser.parse_args()


def _search_ckpt(save_path, input_exp_name):
    """

    :param save_path: /home/xxx/ray_results/ipd1111/
    :param input_exp_name: ipd1111_seed0_iter3
    :param iter_id: 1
    :param seed: 0
    :return: /home/xxx/ray_results/ipd1111/ipd1111_seed0_iter3/PPO_xxxxx
    /checkpoint-100/checkpoint_100
    """

    assert osp.exists(osp.dirname(save_path))

    sub_exp_path = osp.join(save_path,
                            input_exp_name)  # ../exp/exp_seed0_iter3
    if (not osp.exists(save_path)) or \
            (not osp.exists(sub_exp_path)) or \
            (not os.listdir(sub_exp_path)):
        return None

    # should be ['ppo_xxx', .., 'experiment_..']
    dirs = [d for d in os.listdir(sub_exp_path) if osp.isdir(d)]

    if len(dirs) != 1:
        print("We expecte there is only one trial")
        raise ValueError("We expecte there is only one trial")

    trial_dir = dirs[0]
    return get_latest_checkpoint(trial_dir)['path']


def train_one_iteration(
        iter_id, exp_name, init_yaml_path, config, stop_criterion, num_seeds=1,
        num_gpus=0, test_mode=False):
    assert isinstance(iter_id, int)
    assert isinstance(exp_name, str)
    assert isinstance(stop_criterion, dict)
    assert isinstance(init_yaml_path, str)
    assert osp.exists(init_yaml_path)

    local_dir = get_local_dir() if get_local_dir() else "~/ray_results"
    local_dir = os.path.expanduser(local_dir)
    save_path = os.path.join(local_dir, exp_name)
    current_yaml_path = init_yaml_path

    assert 'seed' not in exp_name, exp_name
    assert 'iter' not in exp_name, exp_name

    for i in range(num_seeds):
        input_exp_name = exp_name + "_seed{}_iter{}".format(i, iter_id)

        tmp_config = copy.deepcopy(config)
        tmp_config.update(seed=i)
        tmp_config['env_config']['yaml_path'] = current_yaml_path
        initialize_ray(num_gpus=num_gpus, test_mode=test_mode,
                       local_mode=test_mode)
        tune.run(
            "PPO",
            name=input_exp_name,
            verbose=2 if test_mode else 1,
            local_dir=save_path,
            checkpoint_freq=10,
            checkpoint_at_end=True,
            stop=stop_criterion,
            config=tmp_config
        )

        name_ckpt_mapping = read_yaml(current_yaml_path)
        ckpt_path = _search_ckpt(save_path, input_exp_name)



        last_ckpt_dict = copy.deepcopy(list(name_ckpt_mapping.values())[-1])
        assert isinstance(last_ckpt_dict, dict), last_ckpt_dict
        assert 'path' in last_ckpt_dict, last_ckpt_dict
        last_ckpt_dict.update(path=ckpt_path)

        print("Finish the current last_ckpt_dict: ", last_ckpt_dict)
        name_ckpt_mapping[input_exp_name] = last_ckpt_dict

        current_yaml_path = osp.join(save_path, "post_agent_ppo.yaml")
        out = save_yaml(name_ckpt_mapping, current_yaml_path)
        assert out == current_yaml_path


def test_train_ipd():
    ipd_common_config = {
        "env": IPDEnv,
        "env_config": {
            "env_name": "BipedalWalker-v2",
            "novelty_threshold": None,
            "yaml_path": None
        },
        "callbacks": {"on_episode_end": on_episode_end}
    }

    # algo_config = {"BipedalWalker-v2": {
    #     "num_sgd_iter": 10,
    #     "num_envs_per_worker": 16,
    #     "gamma": 0.99,
    #     "entropy_coeff": 0.001,
    #     "lambda": 0.95,
    #     "lr": 2.5e-4,
    # }}

    algo_config = {"BipedalWalker-v2": {
        "num_sgd_iter": 10,
        "num_envs_per_worker": 1,
        "gamma": 0.99,
        'num_workers': 0,
        "entropy_coeff": 0.001,
        "lambda": 0.95,
        "lr": 2.5e-4,
    }}

    # FIXME this is testing value.
    stop_config = {"BipedalWalker-v2": {
        # "timesteps_total": int(1e7)
        "timesteps_total": 1000,
    }}

    # env_name = args.env
    # FIXME do not hard-coded env_name
    env_name = "BipedalWalker-v2"
    config = algo_config[env_name]
    config.update(ipd_common_config)

    # FIXME add codes here to update novelty_threshold at config.env_config
    # novelty_threshold = None
    novelty_threshold = 0.5
    config['env_config']['novelty_threshold'] = novelty_threshold

    for iter_id in range(3):
        train_one_iteration(
            iter_id, "DELETE_TEST",
            init_yaml_path="../../data/yaml/test-2-agents.yaml",
            config=config,
            stop_criterion=stop_config[env_name],
            test_mode=True)

    print("Finish!")


if __name__ == '__main__':
    test_train_ipd()
