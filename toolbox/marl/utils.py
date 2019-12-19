import logging

import numpy as np
from ray.rllib.policy.sample_batch import MultiAgentBatch

from toolbox.distance import joint_dataset_distance, js_distance

logger = logging.getLogger(__name__)


def _collect_joint_dataset(trainer, worker, sample_size):
    joint_obs = []
    if hasattr(trainer.optimizer, "replay_buffers"):
        # If we are using maddpg, it use ReplayOptimizer, which has this
        # attribute.
        for policy_id, replay_buffer in \
                trainer.optimizer.replay_buffers.items():
            obs = replay_buffer.sample(sample_size)[0]
            joint_obs.append(obs)
    else:
        # If we are using individual PPO, it has no replay buffer,
        # so it seems we have to rollout here to collect the observations

        # Force to collect enough data for us to use.
        tmp_batch = worker.sample()
        count_dict = {k: v.count for k, v in tmp_batch.policy_batches.items()}
        for k in worker.policy_map.keys():
            if k not in count_dict:
                count_dict[k] = 0
        samples = [tmp_batch]
        while any(c < sample_size for c in count_dict.values()):
            tmp_batch = worker.sample()
            for k, v in tmp_batch.policy_batches.items():
                assert k in count_dict, count_dict
                count_dict[k] += v.count
            samples.append(tmp_batch)
        multi_agent_batch = MultiAgentBatch.concat_samples(samples)
        for pid, batch in multi_agent_batch.policy_batches.items():
            batch.shuffle()
            assert batch.count >= sample_size, (
                batch, batch.count,
                [b.count for b in batch.policy_batches.values()]
            )
            joint_obs.append(batch.slice(0, sample_size)['obs'])
    joint_obs = np.concatenate(joint_obs)
    return joint_obs


def on_train_result(info):
    """info only contains trainer and result."""
    sample_size = info['trainer'].config.get("joint_dataset_sample_batch_size")

    if sample_size is None:
        logger.debug(
            "You should specify the value of: "
            "joint_dataset_sample_batch_size in config! Since you "
            "don't set it, we set it to 200 for you."
        )
        sample_size = 200

    # replay_buffers is a dict map policy_id to ReplayBuffer object.
    trainer = info['trainer']
    worker = trainer.workers.local_worker()

    try:
        joint_obs = _collect_joint_dataset(trainer, worker, sample_size)
    except Exception as e:
        logger.info(
            "Encounter error <{}> in on_train_result. Return.".format(e)
        )
        return

    def _replay(policy, pid):
        act, _, infos = policy.compute_actions(joint_obs)
        return pid, act, infos

    ret = {
        pid: [act, infos]
        for pid, act, infos in worker.foreach_policy(_replay)
    }
    # now we have a mapping: policy_id to joint_dataset_replay in 'ret'
    flatten = [act for act, infos in ret.values()]  # flatten action array
    dist_matrix = joint_dataset_distance(flatten)

    mask = np.logical_not(
        np.diag(np.ones(dist_matrix.shape[0])).astype(np.bool)
    )

    flatten_dist = dist_matrix[mask]
    info['result']['distance'] = {}
    info['result']['distance']['overall_mean'] = flatten_dist.mean()
    info['result']['distance']['overall_max'] = flatten_dist.max()
    info['result']['distance']['overall_min'] = flatten_dist.min()
    for i, pid in enumerate(ret.keys()):
        row_without_self = dist_matrix[i][mask[i]]
        info['result']['distance'][pid + "_mean"] = row_without_self.mean()

    tmp_info = ret[pid][1]
    if "behaviour_logits" not in tmp_info:
        # Maybe this is DDPG-like algorithm and the output is not a
        # distribution.
        return

    js_flatten = [infos['behaviour_logits'] for act, infos in ret.values()]
    js_matrix = js_distance(js_flatten)
    flatten_js_dist = js_matrix[mask]
    info['result']['distance_js'] = {}
    info['result']['distance_js']['overall_mean'] = flatten_js_dist.mean()
    info['result']['distance_js']['overall_max'] = flatten_js_dist.max()
    info['result']['distance_js']['overall_min'] = flatten_js_dist.min()
    for i, pid in enumerate(ret.keys()):
        row_without_self = js_matrix[i][mask[i]]
        info['result']['distance_js'][pid + "_mean"] = row_without_self.mean()
