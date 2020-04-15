"""This file provides function to parse a RLLib analysis"""
import logging
import os.path as osp
import pickle
import re
from collections import Iterable

import numpy as np
import pandas as pd
import scipy.interpolate
from ray.tune.analysis.experiment_analysis import Analysis, ExperimentAnalysis
import numbers

def _process_input(path_or_obj):
    if isinstance(path_or_obj, str):
        # analysis directory, pkl file path or experiment state json path.
        path_or_obj = osp.expanduser(path_or_obj)
        assert osp.isfile(path_or_obj) or osp.isdir(path_or_obj)
        if path_or_obj.endswith("pkl"):
            with open(path_or_obj, 'rb') as f:
                trial_dict = pickle.load(f)
        elif path_or_obj.endswith("json"):
            trial_dict = ExperimentAnalysis(path_or_obj).trial_dataframes
        else:
            trial_dict = Analysis(path_or_obj).trial_dataframes
    elif isinstance(path_or_obj, dict):
        trial_dict = path_or_obj
    elif isinstance(path_or_obj, Analysis) or \
            isinstance(path_or_obj, ExperimentAnalysis):
        trial_dict = path_or_obj.trial_dataframes
    else:
        raise NotImplementedError(
            "We expect the input is trial_dataframes dict, an analysis object,"
            " path toward a experiment, path toward a experiment_state json "
            "file.")
    assert isinstance(trial_dict, dict)
    return trial_dict


def _parse_tag(tag):
    """
    tag: 0_normalize_advantage=True,seed=0,tau=0.1,mode=None
    ret: {'normalize_advantage': True, 'seed': 0, 'tau': 0.1, 'mode': None}
    """
    ret = {}
    tag = re.match("\d+_(.*)", tag).groups()[0]
    for part in tag.split(","):
        key, value = part.split('=')
        try:
            value = eval(value)
        except NameError:
            value = value
        ret[key] = value
    return ret


def get_keys(path_or_obj):
    trial_dict = _process_input(path_or_obj)
    keys = set()
    for trial_name, trial_df in trial_dict.items():
        keys.update(_parse_tag(trial_df.experiment_tag[0]).keys())
        keys.update(trial_df.keys())
    return keys


def parse(path_or_obj, interpolate=True, keys=None, name_mapping=None,
          interpolate_x="timesteps_total"):
    """

    :param path_or_obj: can be the following four type of inputs:
        1. the path to the experiment directory which contain experiment_state
        2. the path to the experiment_state json file
        3. a dict object which is the trial_dataframes of an analysis object
        4. an Analysis or ExperimentAnalysis object
    :param interpolate: boolean
    :param keys: the interesting metrics to show, default  is
        all keys, if you use name_mapping, then here should be
        the simplified key of that metric, e.g. num_agents.
    :param name_mapping: map the simplified name of a metric to the real name
        in the Analysis. For example:
            {"num_agents": "env_config/num_agents",
             "vf_loss": "info/learner/agent0/vf_loss"}
    :param interpolate_x: the metric to applied interpolate on, default is
        timesteps_total
    :return: a big pandas dataframe which contains everything.
    """
    # Step 1: Read the data from four possible sources.
    trial_dict = _process_input(path_or_obj)

    if len(trial_dict) == 0:
        print("Empty folder!")
        return pd.DataFrame()

    # Step 2: process the keys that user querying.
    if keys is None:
        # If default, parse all possible keys
        keys = list(next(iter(trial_dict.values())).keys())
        # keys = "episode_reward_mean"
    if isinstance(keys, str):
        keys = [keys]
    assert isinstance(keys, Iterable)
    keys = set(keys)
    keys.add("timesteps_total")
    keys.add("training_iteration")
    investigate_keys = keys.copy()  # use for interpolate

    keys.add("experiment_id")
    for trial_name, trial_df in trial_dict.items():
        tags = _parse_tag(trial_df.experiment_tag[0])
        for tag_name, tag_value in tags.items():
            keys.add(tag_name)
            trial_df[tag_name] = tag_value
            logging.debug(
                "In experiment {}, detect tag {} with value {}".format(
                    trial_df.experiment_tag[0], tag_name, tag_value
                ))
    if name_mapping is not None:
        """To simplify the keys, you can set 
            name_mapping = {
                "env_config/num_agents": "num_agents"
            }
            while in keys, you can simply use "num_agents" as keys.
            """
        assert isinstance(name_mapping, dict)
        new_keys = set()
        for k in keys:
            if k in name_mapping:
                new_keys.add(name_mapping[k])
            else:
                new_keys.add(k)
                name_mapping[k] = k
        keys = new_keys
        reversed_name_mapping = {v: k for k, v in name_mapping.items()}
    logging.info("Current Keys: {}".format(keys))

    # Step 3: Filter the input data with keys
    trial_list = []
    for trial_id, (trial_name, trial_df) in enumerate(trial_dict.items()):
        for k in keys:
            if k not in trial_df:
                logging.info(
                    "<{}> is not in the dataframe of following experiment: {}."
                    " So we fill nan.".format(k, trial_df.experiment_tag[0])
                )
                trial_df[k] = np.nan
        new_trial_df = trial_df[keys]
        assert isinstance(new_trial_df, pd.DataFrame)
        if name_mapping is not None:
            # transform back the keys if once altered.
            new_trial_df = new_trial_df.rename(columns=reversed_name_mapping)
        trial_list.append(new_trial_df)

    # Return if not interpolate the values.
    if not interpolate:
        return pd.concat(trial_list, ignore_index=True)

    # Step 4: conclude the interpolate ticks
    potential = pd.concat([df[interpolate_x] for df in trial_list]).unique()
    potential.sort()
    range_min = 0
    range_max = int(potential.max())
    interpolate_range = np.linspace(
        range_min, range_max, int(max(len(df) for df in trial_list)) * 1
    )

    # Step 5: interpolate for each trail, each key
    new_trial_list = []
    for df in trial_list:
        mask = np.logical_and(
            df[interpolate_x].min() < interpolate_range,
            interpolate_range < df[interpolate_x].max()
        )
        mask_rang = interpolate_range[mask]

        if len(df) > 1:
            new_df = {}
            for k in keys:
                if k in investigate_keys:
                    print(df[k])
                    if isinstance(df[k][0], numbers.Number):
                        new_df[k] = scipy.interpolate.interp1d(
                            df[interpolate_x],
                            df[k]
                        )(mask_rang)
                    else:
                        new_df[k] = df[k].unique()[0]
                else:
                    assert len(df[k].unique()) == 1
                    new_df[k] = df[k].unique()[0]
            new_trial_list.append(pd.DataFrame(new_df))
        else:
            new_trial_list.append(df)

    return pd.concat(new_trial_list, ignore_index=True)
