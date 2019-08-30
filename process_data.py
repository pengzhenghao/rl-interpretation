import datetime
import os
from collections import OrderedDict
from math import floor
from os.path import join

import pandas
import yaml


def get_trial_name(trial_raw_name):
    # The raw name is ugly,
    # like PPO_BipedalWalker-v2_5_seed=25_2019-08-10_16-54-38nmdc0s1h
    # Now we use PPO_BipedalWalker-v2_5_seed=25 as the display name.

    # Remove the date and other codes
    # Results: 'PPO_BipedalWalker-v2_5_seed=25'
    this_year = str(datetime.datetime.now().year)  # "2019"
    delimiter = "_" + this_year  # "_2019"
    name = trial_raw_name.split(delimiter)[0]
    return name


# read data from all json files
def get_trial_data_dict(json_dict):
    trial_data_dict = {}
    for i, (trial_name, json_path) in enumerate(json_dict.items()):
        try:
            dataframe = pandas.read_json(json_path, lines=True)
        except ValueError as err:
            try:
                import json
                ret = []
                with open(json_path, 'r') as f:
                    for line in f:
                        ret.append(json.loads(line))
                    dataframe = pandas.DataFrame(ret)
            except Exception as err:
                print(err)
                continue
        print(
            "[{}/{}] Trial Name: {}\t(Total Iters {})".format(
                i, len(json_dict), trial_name, len(dataframe)
            )
        )
        trial_data_dict[trial_name] = dataframe
    return trial_data_dict


# parse the hierarchy structure of data, clear out the potential EXP
def get_trial_json_dict(exp_name, algo_name, root_dir="~/ray_results"):
    """
    suppose the files looks like:
        - exp-name (e.g. 0811-0to50)
            - PPO_XXX_seed=198_XXX
                - result.json  <= That's what we are parsing.
            - PPO_XXX_seed=199_XXX
            - ...

    json_dict should stores:
    {'PPO_seed=xx': '~/ray_results/exp-name/PPO_seed10/result.json'}

    The user should specified the exp_name, algo_name and root_dir
    """
    input_dir = os.path.abspath(os.path.expanduser(join(root_dir, exp_name)))

    trial_dict = {
        get_trial_name(trial): join(input_dir, trial)
        for trial in os.listdir(input_dir) if trial.startswith(algo_name)
    }

    print(
        "Exp name {}, Algo name {}, Root dir {}. Found {} trials.".format(
            exp_name, algo_name, root_dir, len(trial_dict)
        )
    )

    json_dict = {}
    for name, trial in trial_dict.items():
        json_path = join(trial, "result.json")
        assert os.path.exists(json_path), json_path
        json_dict[name] = json_path
    return json_dict


def get_latest_checkpoint(trial_dir):
    # Input: /home/../ray_results/exp/PPO_xx_xxx/

    ckpt_paths = [
        {
            "path": join(join(trial_dir, ckpt), ckpt.replace("_", "-")),
            "iter": int(ckpt.split('_')[1])
        } for ckpt in os.listdir(trial_dir) if ckpt.startswith("checkpoint")
    ]

    if len(ckpt_paths) == 0:
        return None

    sorted_ckpt_paths = sorted(ckpt_paths, key=lambda pair: pair["iter"])
    return sorted_ckpt_paths[-1]["path"]


def make_ordereddict(list_of_dict, number=None, mode="uniform"):
    assert mode in ['uniform', 'top']
    ret = OrderedDict()
    if number is None:
        number = len(list_of_dict)
    assert number <= len(list_of_dict)

    if mode == 'uniform':
        interval = int(floor(len(list_of_dict) / number))
        # list_of_dict[::interval][-number:]
        # list_of_dict - interval * number

        start_index = len(list_of_dict) % number
        indices = reversed(list_of_dict[:start_index:-interval])
    elif mode == 'top':
        indices = list_of_dict[-number:]
    else:
        raise ValueError()

    for d in indices:
        ret[d['name']] = d['path']
    assert len(ret) == number
    return ret


def get_sorted_trial_ckpt_list(
        sorted_trial_pfm_list, trial_json_dict, get_video_name
):
    """
    Return: [{"name": NAME, "path": CKPT_PATH, ...}, {...}, ...]
    """
    results = []
    for (trial_name, performance) in sorted_trial_pfm_list:
        # varibales show here:
        #    trial_name: PPO_xx_seed=199
        #    json_path: xxx/xxx/trial/result.json
        #    trial_path: xxx/xxx/trial
        json_path = trial_json_dict[trial_name]
        trial_path = os.path.dirname(json_path)
        ckpt = get_latest_checkpoint(trial_path)

        if ckpt is None:
            continue

        cool_name = get_video_name(trial_name, performance)
        results.append(
            {
                "name": cool_name,
                "path": ckpt,
                "performance": float(performance)
            }
        )
    return results


def read_yaml(yaml_path, number=None, mode='top'):
    with open(yaml_path, 'r') as f:
        name_ckpt_list = yaml.safe_load(f)
    name_ckpt_mapping = make_ordereddict(name_ckpt_list, number, mode)
    return name_ckpt_mapping


get_name_ckpt_mapping = read_yaml


def read_batch_yaml(yaml_path_number_dict, mode="top"):
    name_ckpt_mapping = None
    for yaml_path, number in yaml_path_number_dict.items():
        if name_ckpt_mapping is None:
            name_ckpt_mapping = read_yaml(yaml_path, number, mode)
        else:
            name_ckpt_mapping.update(read_yaml(yaml_path, number, mode))
    return name_ckpt_mapping


def generate_yaml(exp_names, algo_name, output_path):
    # Get the trial_name-json_path dict.
    trial_json_dict = {}
    if isinstance(exp_names, str):
        exp_names = [exp_names]
    for exp_name in exp_names:
        trial_json_dict.update(get_trial_json_dict(exp_name, algo_name))
    print("Collected trial_json_dict: ", trial_json_dict)
    # Get the trial_name-trial_data dict. This is not ordered.
    trial_data_dict = get_trial_data_dict(trial_json_dict)
    print("Collected trial_data_dict: ", trial_data_dict)
    K = 3

    trial_performance_list = []

    for i, (trial_name, data) in enumerate(trial_data_dict.items()):
        avg = data["episode_reward_mean"].tail(K).mean()
        trial_performance_list.append([trial_name, avg])

    print("Collected trial_performance_list: ", trial_performance_list)
    sorted_trial_pfm_list = sorted(
        trial_performance_list, key=lambda pair: pair[1]
    )

    def get_video_name(trial_name, performance):
        # trial_name: PPO_BipedalWalker-v2_38_seed=138
        # result: "PPO seed=139 rew=249.01"
        components = trial_name.split("_")
        return "{0} {3} rew={4:.2f}".format(*components, performance)

    results = get_sorted_trial_ckpt_list(
        sorted_trial_pfm_list, trial_json_dict, get_video_name
    )
    with open(output_path, 'w') as f:
        yaml.safe_dump(results, f)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-names", nargs='+', type=str, required=True)
    parser.add_argument("--algo-name", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    args = parser.parse_args()
    assert isinstance(args.exp_names, list) or isinstance(args.exp_names, str)
    assert args.output_path.endswith("yaml")
    ret = generate_yaml(args.exp_names, args.algo_name, args.output_path)
    print("Successfully collect {} agents.".format(len(ret)))
