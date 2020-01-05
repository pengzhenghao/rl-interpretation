import pickle

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.tune.util import merge_dicts

from toolbox.marl.utils import on_train_result as on_train_result_cal_diversity


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


# def on_episode_start(info):
#     episode = info["episode"]
#     episode.user_data["relative_kl"] = {
#         pid: {}
#         for pid in episode._policies.keys()
#     }
#     episode.user_data["unclip_length"] = {
#         pid: {}
#         for pid in episode._policies.keys()
#     }

# def on_postprocess_traj(info):
#     pass

# def on_episode_end(info):
#     episode = info["episode"]
#
#     tmp = "{}-action_kl"
#     for pid, oth in episode.user_data['relative_kl'].items():
#         episode.custom_metrics[tmp.format(pid)] = np.mean(
#             np.concatenate([kl for kl in oth.values()])
#         )
#
#     tmp = "{}-unclipped_ratio"
#     for pid, oth in episode.user_data['unclip_length'].items():
#         if not oth:
#             continue
#         total_length = sum(l for (_, l) in oth.values())
#         unclipped_length = sum(l for (l, _) in oth.values())
#         episode.custom_metrics[tmp.format(pid)] = \
#             unclipped_length / total_length


def _restore_state(ckpt):
    wkload = pickle.load(open(ckpt, 'rb'))['worker']
    state = pickle.loads(wkload)['state']['default_policy']
    return state


def validate_tensor(x, msg=None, enable=True):
    """Validate whether the tensor contain NaN or Inf. Default unable."""
    if enable:
        assert msg is not None
        return tf.check_numerics(x, msg)
    else:
        return x


def assert_nan(arr, enable=True):
    if enable:
        assert np.all(np.isfinite(np.asarray(arr, dtype=np.float32))), arr


def stat(arr, msg="", enable=False):
    if enable:
        print(
            "{}: min {:.5f}, mean {:.5f}, max {:.5f}, std {:.5f}, shape {}, "
            "sum {}, norm {}, mse {}".format(
                msg if msg else "[STAT]", arr.min(), arr.mean(), arr.max(),
                arr.std(), arr.shape, arr.sum(), np.linalg.norm(arr),
                np.mean(np.square(arr))
            )
        )


DIVERSITY_ENCOURAGING = "diversity_encouraging"
USE_BISECTOR = "use_bisector"
USE_DIVERSITY_VALUE_NETWORK = "use_diversity_value_network"
CLIP_DIVERSITY_GRADIENT = "clip_diversity_gradient"
DELAY_UPDATE = "delay_update"
DIVERSITY_REWARD_TYPE = "diversity_reward_type"
REPLAY_VALUES = "replay_values"
TWO_SIDE_CLIP_LOSS = "two_side_clip_loss"
ONLY_TNB = "only_tnb"

I_AM_CLONE = "i_am_clone"
NOVELTY_REWARDS = "novelty_rewards"
NOVELTY_VALUES = "novelty_values"
NOVELTY_ADVANTAGES = "novelty_advantages"
NOVELTY_VALUE_TARGETS = "novelty_value_targets"

dece_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        tau=5e-3,
        callbacks={"on_train_result": on_train_result},
        **{
            DIVERSITY_ENCOURAGING: True,
            USE_BISECTOR: True,
            USE_DIVERSITY_VALUE_NETWORK: True,
            CLIP_DIVERSITY_GRADIENT: True,
            DELAY_UPDATE: True,
            DIVERSITY_REWARD_TYPE: "mse",
            REPLAY_VALUES: True,
            TWO_SIDE_CLIP_LOSS: True,
            I_AM_CLONE: False,
            ONLY_TNB: False,


            # vtrace
            "use_vtrace": False,
            'use_kl_loss': True,
            "clip_rho_threshold": 1.0,  # TODO
            "clip_pg_rho_threshold": 1.0,  # TODO
            "normalize_advantage": True
        }
    )
)
