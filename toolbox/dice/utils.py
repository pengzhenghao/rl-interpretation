import logging

import numpy as np
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.tune.util import merge_dicts

from toolbox.marl.utils import on_train_result as on_train_result_cal_diversity

logger = logging.getLogger(__name__)


def on_train_result(info):
    on_train_result_cal_diversity(info)
    result = info['result']
    if result['custom_metrics']:
        item_list = set()
        for key, val in result['custom_metrics'].items():
            policy_id, item_full_name = key.split('-')
            # item_full_name: action_kl_mean

            item_name = \
                "".join([s + "_" for s in item_full_name.split("_")[:-1]])[:-1]
            # item_name: action_kl
            item_list.add(item_name)

            if item_name not in result:
                result[item_name] = {}

            if policy_id not in result[item_name]:
                result[item_name][policy_id] = {}

            result[item_name][policy_id][item_full_name] = val

        for item_name in item_list:
            result[item_name]['overall_mean'] = np.mean(
                [
                    a[item_name + "_mean"] for a in result[item_name].values()
                    if isinstance(a, dict)
                ]
            )
            result[item_name]['overall_min'] = np.min(
                [
                    a[item_name + "_min"] for a in result[item_name].values()
                    if isinstance(a, dict)
                ]
            )
            result[item_name]['overall_max'] = np.max(
                [
                    a[item_name + "_max"] for a in result[item_name].values()
                    if isinstance(a, dict)
                ]
            )
        result['custom_metrics'].clear()


def on_postprocess_traj(info):
    """We correct the count of the MultiAgentBatch"""
    episode = info['episode']

    if episode._policies[info['agent_id']].config[ONLY_TNB]:
        return

    post_batch = info['post_batch']
    all_pre_batches = info['all_pre_batches']
    agent_id = next(iter(all_pre_batches.keys()))

    if agent_id != info['agent_id']:
        return

    increment_count = int(np.mean(
        [b.count for _, b in all_pre_batches.values()]
    ))
    current_count = episode.batch_builder.count
    corrected_count = current_count - increment_count + post_batch.count
    episode.batch_builder.count = corrected_count


USE_BISECTOR = "use_bisector"
USE_DIVERSITY_VALUE_NETWORK = "use_diversity_value_network"
CLIP_DIVERSITY_GRADIENT = "clip_diversity_gradient"
DELAY_UPDATE = "delay_update"
DIVERSITY_REWARD_TYPE = "diversity_reward_type"
TWO_SIDE_CLIP_LOSS = "two_side_clip_loss"
ONLY_TNB = "only_tnb"

DIVERSITY_REWARDS = "diversity_rewards"
DIVERSITY_VALUES = "diversity_values"
DIVERSITY_ADVANTAGES = "diversity_advantages"
DIVERSITY_VALUE_TARGETS = "diversity_value_targets"
PURE_OFF_POLICY = "pure_off_policy"
NORMALIZE_ADVANTAGE = "normalize_advantage"

dice_default_config = merge_dicts(
    DEFAULT_CONFIG,
    {
        USE_BISECTOR: True,
        USE_DIVERSITY_VALUE_NETWORK: False,
        DELAY_UPDATE: True,
        TWO_SIDE_CLIP_LOSS: True,
        ONLY_TNB: False,
        NORMALIZE_ADVANTAGE: False,

        CLIP_DIVERSITY_GRADIENT: True,
        DIVERSITY_REWARD_TYPE: "mse",
        PURE_OFF_POLICY: False,
        "tau": 5e-3,
        "callbacks": {
            "on_train_result": on_train_result,
            "on_postprocess_traj": on_postprocess_traj
        }
    }

)
