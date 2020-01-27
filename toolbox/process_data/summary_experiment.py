"""This file parse a given trial and report the best reward among all agents
among all time steps."""
import argparse
import copy
import os

import pandas as pd
from ray.tune.analysis import ExperimentAnalysis


def get_experiment_summary(trial_path):
    trial_path = os.path.expanduser(trial_path)
    exp_list = [
        os.path.join(trial_path, p)
        for p in os.listdir(trial_path)
        if p.startswith('experiment_state')
    ]

    exp_path = max(exp_list, key=lambda path: os.path.getmtime(path))

    if len(exp_list) > 1:
        print("We detect more than one experiments in {}. We take {}.".format(
            trial_path, exp_path
        ))

    ana = ExperimentAnalysis(exp_path)
    print("Success fully ")
    trial_dict = {}

    for k, df in ana.trial_dataframes.items():
        r = {}
        for item_name, item in df.iteritems():
            if not item_name.startswith("policy_reward_mean/agent"):
                continue
            r[item_name] = item.max()
        if not r:
            print("We detect this is not a MultiAgent agent. "
                  "Use episode_reward_mean.")
            r['default_policy'] = df.episode_reward_mean.max()
        trial_dict[k] = (max(r, key=lambda k: r[k]), max(r.values()))

    configs = copy.deepcopy(ana.get_all_configs())
    ret_df = []
    for k, v in trial_dict.items():
        config = configs[k].copy()

        for ck, cv in config.items():
            if cv is None:
                config[ck] = 'None'

        config['best_agent_reward'] = v[1]
        config['best_agent_index'] = v[0]
        ret_df.append(config)
    ret_df = pd.DataFrame(ret_df)

    # print the result
    tuning_keys = []
    for item_name, item in ret_df.iteritems():
        if item_name.startswith('best_agent') or item_name == 'seed':
            continue
        try:
            if len(item.unique()) != 1:
                tuning_keys.append(item_name)
        except TypeError:
            pass
    for k, v in ret_df.groupby(tuning_keys):
        print("\n=== Result For: <{}> ===\n".format(k), v.describe())

    info = {'tuning_keys': tuning_keys,
            "trial_dataframes": ana.trial_dataframes}

    return ret_df, info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    ret_df, info = get_experiment_summary(args.path)
