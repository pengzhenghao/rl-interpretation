import datetime
import os
from os.path import join
from collections import OrderedDict
from math import floor

import pandas


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
        except ValueError:
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

    sorted_ckpt_paths = sorted(ckpt_paths, key=lambda pair: pair["iter"])
    return sorted_ckpt_paths[-1]["path"]


def make_ordereddict(list_of_dict, number=None):
    ret = OrderedDict()

    assert number <= len(list_of_dict)
    interval = int(floor(len(list_of_dict) / number))
    # list_of_dict[::interval][-number:]
    # list_of_dict - interval * number

    start_index = len(list_of_dict) % number
    indices = reversed(list_of_dict[:start_index:-interval])

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

        cool_name = get_video_name(trial_name, performance)
        results.append(
            {
                "name": cool_name,
                "path": ckpt,
                "performance": float(performance)
            }
        )

    return results
