import logging

import numpy as np
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae, \
    setup_mixins, kl_and_loss_stats, BEHAVIOUR_LOGITS, Postprocessing, \
    ACTION_LOGP
from ray.rllib.policy.tf_policy import ACTION_PROB

from toolbox.cooperative_exploration.ceppo_debug import on_postprocess_traj, \
    on_episode_start, on_episode_end, on_train_result
from toolbox.cooperative_exploration.ceppo_loss import loss_ceppo
from toolbox.cooperative_exploration.ceppo_postprocess import \
    postprocess_ppo_gae_replay
from toolbox.cooperative_exploration.utils import *
from toolbox.distance import get_kl_divergence
from toolbox.marl.adaptive_extra_loss import merge_dicts, wrap_stats_fn
from toolbox.marl.extra_loss_ppo_trainer import JOINT_OBS, PEER_ACTION, \
    SampleBatch
from toolbox.ppo_es.tnb_es import TNBESTrainer, TNBESPolicy, \
    validate_config as validate_config_tnbes

# from toolbox.ipd.tnb import TNBTrainer, TNBPolicy, \
#     validate_config as validate_config_tnbes

logger = logging.getLogger(__name__)

# FIXME think twice on the config design
cetnb_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        # learn_with_peers=True,
        # use_joint_dataset=False,

        mode=REPLAY_VALUES,
        clip_action_prob_kl=1,
        clip_action_prob_ratio=1,
        # clip_advantage=False,

        # check_nan=True,
        # clip_action_prob=0.5,  # DEPRECATED, +- 150% is allowed

        # Don't touch
        callbacks={
            "on_train_result": on_train_result,
            "on_episode_start": on_episode_start,
            "on_postprocess_traj": on_postprocess_traj,
            "on_episode_end": on_episode_end
        }
    )
)


def _compute_logp(logit, x):
    # Only for DiagGaussian distribution. Copied from tf_action_dist.py
    logit = logit.astype(np.float64)
    x = np.expand_dims(x.astype(np.float64), 1) if x.ndim == 1 else x
    mean, log_std = np.split(logit, 2, axis=1)
    logp = (
            -0.5 * np.sum(np.square((x - mean) / np.exp(log_std)), axis=1) -
            0.5 * np.log(2.0 * np.pi) * x.shape[1] - np.sum(log_std, axis=1)
    )
    p = np.exp(logp)
    return logp, p


def _clip_batch(other_batch, clip_action_prob_kl):
    kl = get_kl_divergence(
        source=other_batch[BEHAVIOUR_LOGITS],
        target=other_batch["other_logits"],
        mean=False
    )

    mask = kl < clip_action_prob_kl
    length = len(mask)
    info = {"kl": kl, "unclip_length": length, "length": len(mask)}

    if not np.all(mask):
        length = mask.argmin()
        info['unclip_length'] = length
        if length == 0:
            return None, info
        assert length < len(other_batch['action_logp'])
        other_batch = other_batch.slice(0, length)

    return other_batch, info


def postprocess_ceppo(policy, sample_batch, others_batches=None, episode=None):
    if not policy.loss_initialized():
        batch = postprocess_ppo_gae(policy, sample_batch)
        batch["advantages_unnormalized"] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch['debug_ratio'] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch['debug_fake_adv'] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        if policy.config[DIVERSITY_ENCOURAGING] or policy.config[CURIOSITY]:
            # assert not policy.config["use_joint_dataset"]
            batch[JOINT_OBS] = np.zeros_like(
                sample_batch[SampleBatch.CUR_OBS], dtype=np.float32
            )
            batch[PEER_ACTION] = np.zeros_like(
                sample_batch[SampleBatch.ACTIONS], dtype=np.float32
            )
        return batch

    batch = sample_batch

    if policy.config[REPLAY_VALUES]:
        # a little workaround. We normalize advantage for all batch before
        # concatnation.
        tmp_batch = postprocess_ppo_gae(policy, batch)
        value = tmp_batch[Postprocessing.ADVANTAGES]
        standardized = (value - value.mean()) / max(1e-4, value.std())
        tmp_batch[Postprocessing.ADVANTAGES] = standardized
        batches = [tmp_batch]
    else:
        batches = [postprocess_ppo_gae(policy, batch)]

    for pid, (other_policy, other_batch_raw) in others_batches.items():
        # The logic is that EVEN though we may use DISABLE or NO_REPLAY_VALUES,
        # but we still want to take a look of those statics.
        # Maybe in the future we can add knob to remove all such slowly stats.

        if other_batch_raw is None:
            continue

        other_batch = other_batch_raw.copy()

        # four fields that we will overwrite.
        # Two additional: advantages / value target
        other_batch["other_action_logp"] = other_batch[ACTION_LOGP].copy()
        other_batch["other_action_prob"] = other_batch[ACTION_PROB].copy()
        other_batch["other_logits"] = other_batch[BEHAVIOUR_LOGITS].copy()
        other_batch["other_vf_preds"] = other_batch[SampleBatch.VF_PREDS
        ].copy()

        # use my policy to evaluate the values and other relative data
        # of other's samples.
        replay_result = policy.compute_actions(
            other_batch[SampleBatch.CUR_OBS]
        )[2]

        other_batch[SampleBatch.VF_PREDS] = replay_result[SampleBatch.VF_PREDS]
        other_batch[BEHAVIOUR_LOGITS] = replay_result[BEHAVIOUR_LOGITS]

        other_batch[ACTION_LOGP], other_batch[ACTION_PROB] = \
            _compute_logp(
                other_batch[BEHAVIOUR_LOGITS],
                other_batch[SampleBatch.ACTIONS]
            )

        if policy.config[DISABLE]:
            continue
        elif not policy.config[REPLAY_VALUES]:
            batches.append(postprocess_ppo_gae(policy, other_batch_raw))
        else:  # replay values
            if other_batch is not None:  # it could be None due to clipping.
                batches.append(
                    postprocess_ppo_gae_replay(
                        policy, other_batch, other_policy
                    )
                )

    for batch in batches:
        batch[Postprocessing.ADVANTAGES + "_unnormalized"] = batch[
            Postprocessing.ADVANTAGES].copy().astype(np.float32)
        if "debug_ratio" not in batch:
            assert "debug_fake_adv" not in batch
            batch['debug_fake_adv'] = batch['debug_ratio'] = np.zeros_like(
                batch['advantages'], dtype=np.float32
            )

    return SampleBatch.concat_samples(batches) if len(batches) != 1 \
        else batches[0]


def setup_mixins_ceppo(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)


def wrap_stats_ceppo(policy, train_batch):
    if policy.config[DIVERSITY_ENCOURAGING]:
        return wrap_stats_fn(policy, train_batch)
    ret = kl_and_loss_stats(policy, train_batch)
    if hasattr(policy.loss_obj, "stats"):
        assert isinstance(policy.loss_obj.stats, dict)
        ret.update(policy.loss_obj.stats)
    return ret


# def wrap_after_train_result_ceppo(trainer, fetches):
#     if trainer.config[DIVERSITY_ENCOURAGING] or trainer.config[CURIOSITY]:
#         wrap_after_train_result(trainer, fetches)
#     else:
#         update_kl(trainer, fetches)


CETNBPolicy = TNBESPolicy.with_updates(
    name="CETNBPolicy",
    get_default_config=lambda: cetnb_default_config,
    postprocess_fn=postprocess_ceppo,
    loss_fn=loss_ceppo,
)


# FIXME the key modification is the loss. We introduce new element such as
#  the 'other_values' in postprocess. Then we generate advantage based on them.
#  We then use the


def validate_config(config):
    validate_config_tnbes(config)

    mode = config['mode']
    # FIXME think twice on the MODES.
    if mode not in [DISABLE, DIVERSITY_ENCOURAGING_DISABLE,
                    CURIOSITY_DISABLE,
                    CURIOSITY_KL_DISABLE]:
        num_agents = len(config['multiagent']['policies'])
        config['train_batch_size'] = int(
            config['train_batch_size'] // num_agents
        )

        config['num_envs_per_worker'] = max(
            1, int(config['num_envs_per_worker'] // num_agents)
        )

        if config['train_batch_size'] < config["sgd_minibatch_size"]:
            raise ValueError(
                "You are using too many agents here! Current"
                " train_batch_size {}, sgd_minibatch_size {},"
                " num_agents {}.".format(
                    config['train_batch_size'], config[
                        "sgd_minibatch_size"],
                    num_agents
                )
            )


CETNBTrainer = TNBESTrainer.with_updates(
    name="CETNBTrainer",
    default_config=cetnb_default_config,
    default_policy=CETNBPolicy,
    validate_config=validate_config
)
# FIXME So till now the only change to TNBESTrainer
