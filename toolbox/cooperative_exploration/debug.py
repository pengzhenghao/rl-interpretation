import numpy as np
import tensorflow as tf

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
            result[item_name]['over_all_mean'] = np.mean([
                a[item_name + "_mean"] for a in result[item_name].values()
                if isinstance(a, dict)
            ])
            result[item_name]['over_all_min'] = np.min([
                a[item_name + "_min"] for a in result[item_name].values()
                if isinstance(a, dict)
            ])
            result[item_name]['over_all_max'] = np.max([
                a[item_name + "_max"] for a in result[item_name].values()
                if isinstance(a, dict)
            ])

        result['custom_metrics'].clear()


def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["relative_kl"] = {
        pid: {} for pid in episode._policies.keys()
    }
    episode.user_data["unclip_length"] = {
        pid: {} for pid in episode._policies.keys()
    }


def on_postprocess_traj(info):
    pass


def on_episode_end(info):
    episode = info["episode"]

    tmp = "{}-action_kl"
    for pid, oth in episode.user_data['relative_kl'].items():
        episode.custom_metrics[tmp.format(pid)] = np.mean(
            [kl for kl in oth.values()]
        )

    tmp = "{}-unclipped_ratio"
    for pid, oth in episode.user_data['unclip_length'].items():
        total_length = sum(l for (_, l) in oth.values())
        unclipped_length = sum(l for (l, _) in oth.values())
        episode.custom_metrics[tmp.format(
            pid)] = unclipped_length / total_length


def validate_tensor(x, msg=None, enable=False):
    """Validate whether the tensor contain NaN or Inf. Default unable."""
    if enable:
        assert msg is not None
        return tf.check_numerics(x, msg)
    else:
        return x
