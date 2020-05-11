"""
This files defines some constants and provides some useful utilities.
"""
import logging

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_DEFAULT
from ray.rllib.utils import try_import_tf

from toolbox.marl.utils import on_train_result as on_train_result_cal_diversity
from toolbox.utils import merge_dicts

tf = try_import_tf()
logger = logging.getLogger(__name__)


class DiCECallbacks(DefaultCallbacks):
    def on_postprocess_trajectory(self, *args, **kwargs):
        on_postprocess_trajectory(*args, **kwargs)

    def on_train_result(self, *args, **kwargs):
        on_train_result(*args, **kwargs)


def on_train_result(trainer, result, **_):
    on_train_result_cal_diversity(dict(trainer=trainer, result=result))
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
                    a[item_name + "_mean"] for a in
                    result[item_name].values()
                    if isinstance(a, dict)
                ]
            )
            result[item_name]['overall_min'] = np.min(
                [
                    a[item_name + "_min"] for a in
                    result[item_name].values()
                    if isinstance(a, dict)
                ]
            )
            result[item_name]['overall_max'] = np.max(
                [
                    a[item_name + "_max"] for a in
                    result[item_name].values()
                    if isinstance(a, dict)
                ]
            )
        result['custom_metrics'].clear()


def on_postprocess_trajectory(worker, episode, agent_id, policy_id,
                              policies, postprocessed_batch, original_batches,
                              **_):
    """Originally, the sampled steps is accumulated by the count of the
    batch
    return by the postprocess function in dice_postprocess.py. Since we
    merge the batches of other polices into the batch of one policy,
    the count of the batch should changed to the number of sampled of all
    policies. However it's not changed. In this function, we correct the
    count of the MultiAgentBatch in order to make a fair comparison between
    DiCE and baseline.
    """
    if episode._policies[agent_id].config[ONLY_TNB]:
        # ONLY_TNB modes mean we using purely DR, without CE. So in that
        # case
        # we don't need to correct the count since no other batches are
        # merged.
        return

    if agent_id != next(iter(original_batches.keys())):
        return

    increment_count = int(
        np.mean([b.count for _, b in original_batches.values()])
    )
    current_count = episode.batch_builder.count
    corrected_count = current_count - increment_count + \
                      postprocessed_batch.count
    episode.batch_builder.count = corrected_count


USE_BISECTOR = "use_bisector"  # If false, the the DR is disabled.
USE_DIVERSITY_VALUE_NETWORK = "use_diversity_value_network"
DELAY_UPDATE = "delay_update"
TWO_SIDE_CLIP_LOSS = "two_side_clip_loss"
ONLY_TNB = "only_tnb"  # If true, then the CE is disabled.

CLIP_DIVERSITY_GRADIENT = "clip_diversity_gradient"
DIVERSITY_REWARD_TYPE = "diversity_reward_type"
DIVERSITY_REWARDS = "diversity_rewards"
DIVERSITY_VALUES = "diversity_values"
DIVERSITY_ADVANTAGES = "diversity_advantages"
DIVERSITY_VALUE_TARGETS = "diversity_value_targets"
PURE_OFF_POLICY = "pure_off_policy"
NORMALIZE_ADVANTAGE = "normalize_advantage"

dice_default_config = merge_dicts(
    PPO_DEFAULT,
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
        "vf_ratio_clip_param": 0.05,  # Not pass to PPOLossTwoSideClip
        "callbacks": DiCECallbacks,
        "grad_clip": 10.0
    }
)


def get_kl_divergence(source, target, mean=True):
    assert source.ndim == 2
    assert target.ndim == 2

    source_mean, source_log_std = np.split(source, 2, axis=1)
    target_mean, target_log_std = np.split(target, 2, axis=1)

    kl_divergence = np.sum(
        target_log_std - source_log_std + (
                np.square(np.exp(source_log_std)) +
                np.square(source_mean - target_mean)
        ) / (2.0 * np.square(np.exp(target_log_std)) + 1e-10) - 0.5,
        axis=1
    )
    kl_divergence = np.clip(kl_divergence, 1e-12, 1e38)  # to avoid inf
    if mean:
        averaged_kl_divergence = np.mean(kl_divergence)
        return averaged_kl_divergence
    else:
        return kl_divergence
