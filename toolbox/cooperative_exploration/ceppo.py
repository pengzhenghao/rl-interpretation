import logging

import numpy as np
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, validate_config, \
    update_kl
from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae, \
    setup_mixins, kl_and_loss_stats, BEHAVIOUR_LOGITS, Postprocessing, \
    ACTION_LOGP
from ray.rllib.policy.tf_policy import ACTION_PROB
from ray.tune.registry import _global_registry, ENV_CREATOR

from toolbox.cooperative_exploration.ceppo_loss import loss_ceppo
from toolbox.cooperative_exploration.ceppo_debug import on_postprocess_traj, \
    on_episode_start, on_episode_end, on_train_result, \
    assert_nan
from toolbox.cooperative_exploration.ceppo_postprocess import \
    postprocess_ppo_gae_replay
from toolbox.cooperative_exploration.utils import *
from toolbox.distance import get_kl_divergence
from toolbox.marl.adaptive_extra_loss import AdaptiveExtraLossPPOTrainer, \
    AdaptiveExtraLossPPOTFPolicy, merge_dicts, NoveltyParamMixin, mixin_list, \
    AddLossMixin, wrap_stats_fn, wrap_after_train_result
from toolbox.marl.extra_loss_ppo_trainer import JOINT_OBS, PEER_ACTION, \
    SampleBatch, get_cross_policy_object
from toolbox.modified_rllib.multi_gpu_optimizer import \
    LocalMultiGPUOptimizerModified

logger = logging.getLogger(__name__)

ceppo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        learn_with_peers=True,
        use_joint_dataset=False,
        mode=REPLAY_VALUES,
        clip_action_prob_kl=1,
        clip_action_prob_ratio=1,
        clip_advantage=False,
        check_nan=True,
        # clip_action_prob=0.5,  # DEPRECATED, +- 150% is allowed
        callbacks={
            "on_train_result": on_train_result,
            "on_episode_start": on_episode_start,
            "on_postprocess_traj": on_postprocess_traj,
            "on_episode_end": on_episode_end
        }
    )
)


def validate_and_rewrite_config(config):
    assert config["clip_action_prob_ratio"] == 1

    mode = config['mode']
    assert mode in OPTIONAL_MODES

    # fill multiagent automatically in config
    assert _global_registry.contains(ENV_CREATOR, config["env"])
    env_creator = _global_registry.get(ENV_CREATOR, config["env"])
    tmp_env = env_creator(config["env_config"])
    config["multiagent"]["policies"] = {
        i: (None, tmp_env.observation_space, tmp_env.action_space, {})
        for i in tmp_env.agent_ids
    }
    config["multiagent"]["policy_mapping_fn"] = lambda x: x

    # hyper-parameter: DIVERSITY_ENCOURAGING
    if mode in [DIVERSITY_ENCOURAGING, DIVERSITY_ENCOURAGING_NO_RV,
                DIVERSITY_ENCOURAGING_DISABLE,
                DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND]:
        config[DIVERSITY_ENCOURAGING] = True
        config.update(
            # we set increment to zero, to make the novelty_target fix to the
            # initial value.
            novelty_loss_increment=0,
            # novelty_loss_increment=10.0,
            novelty_loss_param_init=0.000001,
            novelty_loss_running_length=10,
            joint_dataset_sample_batch_size=200,
            novelty_mode="mean",
            use_joint_dataset=False
        )
    else:
        config[DIVERSITY_ENCOURAGING] = False

    # hyper-parameter: REPLAY_VALUES
    if mode in [REPLAY_VALUES, DIVERSITY_ENCOURAGING, CURIOSITY, CURIOSITY_KL]:
        config[REPLAY_VALUES] = True
    else:
        config[REPLAY_VALUES] = False

    # hyper-parameter: CURIOSITY
    if mode in [CURIOSITY, CURIOSITY_NO_RV, CURIOSITY_DISABLE,
                CURIOSITY_DISABLE_AND_EXPAND, CURIOSITY_KL, CURIOSITY_KL_NO_RV,
                CURIOSITY_KL_DISABLE, CURIOSITY_KL_DISABLE_AND_EXPAND]:
        config[CURIOSITY] = True

        config.update(
            novelty_loss_increment=0,
            novelty_loss_param_init=0.000001,
            novelty_loss_running_length=10,
            joint_dataset_sample_batch_size=200,
            novelty_mode="mean",
            use_joint_dataset=False
        )
        # if "curiosity_tensity" not in config:
        # config["curiosity_tensity"] = 0.1  # can be tuned, if you wish.
        if mode in [CURIOSITY, CURIOSITY_NO_RV, CURIOSITY_DISABLE,
                    CURIOSITY_DISABLE_AND_EXPAND]:
            config['curiosity_type'] = 'mse'
        elif mode in [CURIOSITY_KL, CURIOSITY_KL_NO_RV, CURIOSITY_KL_DISABLE,
                      CURIOSITY_KL_DISABLE_AND_EXPAND]:
            config['curiosity_type'] = 'kl'
    else:
        config[CURIOSITY] = False

    # hyper-parameter: DISABLE
    if mode in [DISABLE, DISABLE_AND_EXPAND, DIVERSITY_ENCOURAGING_DISABLE,
                DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND, CURIOSITY_DISABLE,
                CURIOSITY_DISABLE_AND_EXPAND, CURIOSITY_KL_DISABLE,
                CURIOSITY_KL_DISABLE_AND_EXPAND]:
        config[DISABLE] = True
    else:
        config[DISABLE] = False

    # Update 20191211: Instead of expand train batch for all modes, we want to
    #  shrink train batch for all modes except DISABLE.
    if mode not in [DISABLE, DIVERSITY_ENCOURAGING_DISABLE, CURIOSITY_DISABLE,
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
                    config['train_batch_size'], config["sgd_minibatch_size"],
                    num_agents
                )
            )

        # config['num_envs_per_worker'] = \
        #     config['num_envs_per_worker'] * num_agents

    # The below codes is used before Update 20191211.
    # DISABLE_AND_EXPAND requires to modified the config.
    # if mode in [DISABLE_AND_EXPAND, DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND,
    #             CURIOSITY_DISABLE_AND_EXPAND,
    #             CURIOSITY_KL_DISABLE_AND_EXPAND]:
    #     num_agents = len(config['multiagent']['policies'])
    #     config['train_batch_size'] = config['train_batch_size'] * num_agents
    #     config['num_envs_per_worker'] = \
    #         config['num_envs_per_worker'] * num_agents

    # validate config
    validate_config(config)
    assert not config.get("use_joint_dataset")
    assert "callbacks" in config
    assert "on_train_result" in config['callbacks']
    assert DISABLE in config
    assert DIVERSITY_ENCOURAGING in config
    assert REPLAY_VALUES in config
    assert CURIOSITY in config
    if config[CURIOSITY]:
        assert "curiosity_type" in config
        # assert "curiosity_tensity" in config
        assert config["curiosity_type"] in ["kl", "mse"]
    else:
        assert "curiosity_type" not in config
        # assert "curiosity_tensity" not in config


def _add_intrinsic_reward(policy, my_batch, others_batches, config):
    # if using mse
    OBS = SampleBatch.CUR_OBS
    my_rew = my_batch[SampleBatch.REWARDS]

    replays = []
    for (other_policy, _) in others_batches.values():
        _, _, info = other_policy.compute_actions(my_batch[OBS])
        replays.append(info[BEHAVIOUR_LOGITS])

    if not replays:
        return my_batch

    if config["curiosity_type"] == "kl":
        intrinsic_reward = np.mean(
            [
                get_kl_divergence(
                    my_batch[BEHAVIOUR_LOGITS], logit, mean=False
                ) for logit in replays
            ],
            axis=0
        )

    elif config["curiosity_type"] == "mse":
        replays = [np.split(logit, 2, axis=1)[0] for logit in replays]

        my_act = np.split(my_batch[BEHAVIOUR_LOGITS], 2, axis=1)[0]
        intrinsic_reward = np.mean(
            [(np.square(my_act - other_act)).mean(1) for other_act in replays],
            axis=0
        )

    else:
        raise NotImplementedError(
            "Wrong curiosity type! {} not in ['kl', 'mse']".format(
                config['curiosity_type']
            )
        )

    # normalize
    intrinsic_reward = (intrinsic_reward - intrinsic_reward.min(
    )) / (intrinsic_reward.max() - intrinsic_reward.min() + 1e-12)
    intrinsic_reward = intrinsic_reward * (my_rew.max() - my_rew.min())
    intrinsic_reward = intrinsic_reward * policy.novelty_loss_param_val

    new_rew = my_rew + intrinsic_reward
    assert new_rew.ndim == 1
    my_batch[SampleBatch.REWARDS] = new_rew

    # we omit the intrinsic reward of the first "prev_state", this will
    # harm a little bit precision but, ..., never mind.
    new_prev_rew = my_batch[SampleBatch.PREV_REWARDS]
    new_prev_rew[1:] = new_rew[:-1]
    my_batch[SampleBatch.PREV_REWARDS] = new_prev_rew

    return my_batch


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

    # ratio = np.exp(
    #     other_batch['action_logp'] - other_batch["other_action_logp"])

    # mask = np.logical_and(ratio < 1 + clip_action_prob,
    #                       ratio > q1 - clip_action_prob)

    # length means the length of the unclipped trajectory

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
            assert not policy.config["use_joint_dataset"]
            batch[JOINT_OBS] = np.zeros_like(
                sample_batch[SampleBatch.CUR_OBS], dtype=np.float32
            )
            batch[PEER_ACTION] = np.zeros_like(
                sample_batch[SampleBatch.ACTIONS], dtype=np.float32
            )
        return batch

    if policy.config[CURIOSITY]:
        batch = _add_intrinsic_reward(
            policy, sample_batch, others_batches, policy.config
        )
    else:
        batch = sample_batch

    # if policy.config[DISABLE]:
    #     # Disable for not adding other's info in my batch.
    #     batch = postprocess_ppo_gae(policy, batch)
    #     batch[Postprocessing.ADVANTAGES + "_unnormalized"] = batch[
    #         Postprocessing.ADVANTAGES].copy().astype(np.float32)
    #     if "debug_ratio" not in batch:
    #         batch['debug_ratio'] = np.ones_like(batch['advantages'],
    #                                             dtype=np.float32)
    #     if "debug_fake_adv" not in batch:
    #         batch['debug_fake_adv'] = np.ones_like(batch['advantages'],
    #                                                dtype=np.float32)
    #     return batch

    my_id = "agent{}".format(sample_batch['agent_index'][0])

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

        # TODO(pengzh) it's ok to delete these two data in batch.
        other_batch[ACTION_LOGP], other_batch[ACTION_PROB] = \
            _compute_logp(
                other_batch[BEHAVIOUR_LOGITS],
                other_batch[SampleBatch.ACTIONS]
            )

        if policy.config["clip_action_prob_kl"] is not None:
            other_batch, info = _clip_batch(
                other_batch, policy.config["clip_action_prob_kl"]
            )
            episode.user_data['relative_kl'][my_id][pid] = info['kl']
            episode.user_data['unclip_length'][my_id][pid] = (
                info['unclip_length'], info['length']
            )

        if policy.config['check_nan'] and (other_batch is not None):
            assert other_batch[SampleBatch.CUR_OBS].ndim == 2, other_batch
            assert other_batch[BEHAVIOUR_LOGITS].ndim == 2
            assert_nan(other_batch[SampleBatch.CUR_OBS])
            assert_nan(other_batch[BEHAVIOUR_LOGITS])
            assert_nan(other_batch[ACTION_LOGP])
            assert_nan(other_batch[ACTION_PROB])

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
    if config[DIVERSITY_ENCOURAGING] or config[CURIOSITY]:
        AddLossMixin.__init__(policy, config)
        NoveltyParamMixin.__init__(policy, config)


def cross_policy_object_without_joint_dataset(
        multi_agent_batch, self_optimizer
):
    """Modification: when not using DISABLE, the data of each agents is
    identical, since we need Cooperative Exploration. So when computing the
    joint dataset here for the following diversity-encouraging, we need to
    carefully choose the things we use for replay.
    """
    config = self_optimizer.workers._remote_config
    if config[DISABLE]:
        return get_cross_policy_object(multi_agent_batch, self_optimizer)

    # In that cases, the batch for all agents are the same: all of them
    # are a combination of all of them. So we do not need to replay
    # for each agent's batch (since they are identical).
    return_dict = {}
    local_worker = self_optimizer.workers.local_worker()
    if len(set(b.count
               for b in multi_agent_batch.policy_batches.values())) != 1:
        msg = "We detected the multi_agent_batch has different length of " \
              "batches in its policy_batches: length {}.".format(
            {k: b.count for k, b in multi_agent_batch.policy_batches.items()}
        )
        print(msg)
        logger.warning(msg)
    joint_obs = next(iter(multi_agent_batch.policy_batches.values()))['obs']

    def _replay(policy, replay_pid):
        act, _, infos = policy.compute_actions(joint_obs)
        return replay_pid, act

    for pid, act in local_worker.foreach_policy(_replay):
        return_dict[pid] = act
    return {JOINT_OBS: joint_obs, PEER_ACTION: return_dict}


def choose_policy_optimizer_modified(workers, config):
    """The original optimizer has wrong number of trained samples stats.
    So we make little modification and use the corrected optimizer."""
    if config["simple_optimizer"]:
        raise NotImplementedError()
        # return SyncSamplesOptimizer(
        #     workers,
        #     num_sgd_iter=config["num_sgd_iter"],
        #     train_batch_size=config["train_batch_size"],
        #     sgd_minibatch_size=config["sgd_minibatch_size"],
        #     standardize_fields=["advantages"]
        # )

    num_agents = len(config['multiagent']['policies'])

    if config[DISABLE]:
        compute_num_steps_sampled = None
    else:

        def compute_num_steps_sampled(batch):
            counts = np.mean([b.count for b in batch.policy_batches.values()])
            return int(counts / num_agents)

    if config[DIVERSITY_ENCOURAGING] or config[CURIOSITY]:
        process_multiagent_batch_fn = cross_policy_object_without_joint_dataset
        no_split_list = [PEER_ACTION, JOINT_OBS]
    else:
        process_multiagent_batch_fn = None
        no_split_list = None

    return LocalMultiGPUOptimizerModified(
        workers,
        compute_num_steps_sampled=compute_num_steps_sampled,
        no_split_list=no_split_list,
        process_multiagent_batch_fn=process_multiagent_batch_fn,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=["advantages"]
        if not config['clip_advantage'] else [],
        shuffle_sequences=config["shuffle_sequences"]
    )


def wrap_stats_ceppo(policy, train_batch):
    if policy.config[DIVERSITY_ENCOURAGING]:
        return wrap_stats_fn(policy, train_batch)
    ret = kl_and_loss_stats(policy, train_batch)
    if hasattr(policy.loss_obj, "stats"):
        assert isinstance(policy.loss_obj.stats, dict)
        ret.update(policy.loss_obj.stats)
    if policy.config[CURIOSITY]:
        ret.update(
            novelty_loss_param=policy.novelty_loss_param,
            novelty_target=policy.novelty_target_tensor,
            novelty_loss=policy.novelty_loss
        )
    return ret


def wrap_after_train_result_ceppo(trainer, fetches):
    if trainer.config[DIVERSITY_ENCOURAGING] or trainer.config[CURIOSITY]:
        wrap_after_train_result(trainer, fetches)
    else:
        update_kl(trainer, fetches)


CEPPOTFPolicy = AdaptiveExtraLossPPOTFPolicy.with_updates(
    name="CEPPOTFPolicy",
    get_default_config=lambda: ceppo_default_config,
    postprocess_fn=postprocess_ceppo,
    loss_fn=loss_ceppo,
    before_loss_init=setup_mixins_ceppo,
    stats_fn=wrap_stats_ceppo,
    mixins=mixin_list + [AddLossMixin, NoveltyParamMixin]
)

CEPPOTrainer = AdaptiveExtraLossPPOTrainer.with_updates(
    name="CEPPO",
    after_optimizer_step=wrap_after_train_result_ceppo,
    default_config=ceppo_default_config,
    default_policy=CEPPOTFPolicy,
    validate_config=validate_and_rewrite_config,
    make_policy_optimizer=choose_policy_optimizer_modified
)
