import logging

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, validate_config, \
    SyncSamplesOptimizer, update_kl
from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae, \
    make_tf_callable, setup_mixins, kl_and_loss_stats, ppo_surrogate_loss, \
    BEHAVIOUR_LOGITS
from ray.tune.registry import _global_registry, ENV_CREATOR

from toolbox.distance import get_kl_divergence
from toolbox.marl.adaptive_extra_loss import AdaptiveExtraLossPPOTrainer, \
    AdaptiveExtraLossPPOTFPolicy, merge_dicts, NoveltyParamMixin, mixin_list, \
    AddLossMixin, wrap_stats_fn, wrap_after_train_result
from toolbox.marl.extra_loss_ppo_trainer import JOINT_OBS, PEER_ACTION, \
    SampleBatch, extra_loss_ppo_loss, get_cross_policy_object, novelty_loss_mse
from toolbox.marl.utils import on_train_result
from toolbox.modified_rllib.multi_gpu_optimizer import \
    LocalMultiGPUOptimizerModified

logger = logging.getLogger(__name__)

DISABLE = "disable"  # also serve as config's key
DISABLE_AND_EXPAND = "disable_and_expand"
REPLAY_VALUES = "replay_values"  # also serve as config's key
NO_REPLAY_VALUES = "no_replay_values"

DIVERSITY_ENCOURAGING = "diversity_encouraging"  # also serve as config's key
DIVERSITY_ENCOURAGING_NO_RV = "diversity_encouraging_without_replay_values"
DIVERSITY_ENCOURAGING_DISABLE = "diversity_encouraging_disable"
DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND = \
    "diversity_encouraging_disable_and_expand"

CURIOSITY = "curiosity"  # also serve as config's key
CURIOSITY_NO_RV = "curiosity_without_replay_values"
CURIOSITY_DISABLE = "curiosity_disable"
CURIOSITY_DISABLE_AND_EXPAND = "curiosity_disable_and_expand"

CURIOSITY_KL = "curiosity_KL"  # also serve as config's key
CURIOSITY_KL_NO_RV = "curiosity_KL_without_replay_values"
CURIOSITY_KL_DISABLE = "curiosity_KL_disable"
CURIOSITY_KL_DISABLE_AND_EXPAND = "curiosity_KL_disable_and_expand"

OPTIONAL_MODES = [
    DISABLE, DISABLE_AND_EXPAND, REPLAY_VALUES, NO_REPLAY_VALUES,
    DIVERSITY_ENCOURAGING, DIVERSITY_ENCOURAGING_NO_RV,
    DIVERSITY_ENCOURAGING_DISABLE, DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND,
    CURIOSITY, CURIOSITY_NO_RV, CURIOSITY_DISABLE,
    CURIOSITY_DISABLE_AND_EXPAND, CURIOSITY_KL, CURIOSITY_KL_NO_RV,
    CURIOSITY_KL_DISABLE, CURIOSITY_KL_DISABLE_AND_EXPAND
]

ceppo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        learn_with_peers=True,
        use_joint_dataset=False,
        mode=REPLAY_VALUES,
        callbacks={"on_train_result": on_train_result}
    )
    # you should add {"on_train_result": on_train_result} to callbacks.
)


def validate_and_rewrite_config(config):
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
    mean, log_std = np.split(logit, 2, axis=1)
    logp = (-0.5 * np.sum(
        np.square((x - mean) / np.exp(log_std)), axis=1) -
            0.5 * np.log(2.0 * np.pi) * x.shape[1] -
            np.sum(log_std, axis=1))
    p = np.exp(logp)
    return logp, p


def postprocess_ceppo(policy, sample_batch, others_batches=None, episode=None):
    if not policy.loss_initialized():
        batch = postprocess_ppo_gae(policy, sample_batch)
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

    if policy.config[DISABLE]:
        # Disable for not adding other's info in my batch.
        return postprocess_ppo_gae(policy, batch)

    batches = [postprocess_ppo_gae(policy, batch)]
    for pid, (_, batch) in others_batches.items():
        if policy.config[REPLAY_VALUES]:
            # use my policy to evaluate the values of other's samples.
            # batch[SampleBatch.VF_PREDS] = policy._value_batch(
            #     batch[SampleBatch.CUR_OBS], batch[SampleBatch.PREV_ACTIONS],
            #     batch[SampleBatch.PREV_REWARDS]
            # )  # changing VF_PREDS will change VALUE_TARGET and ADVANTAGE

            # Except values, we also need to replay the following data.
            replay_result = policy.compute_actions(batch[SampleBatch.CUR_OBS])
            batch[SampleBatch.VF_PREDS] = replay_result[2]['vf_preds']
            batch[BEHAVIOUR_LOGITS] = replay_result[2]['behaviour_logits']
            batch["action_logp"], batch["action_prob"] = _compute_logp(
                batch[BEHAVIOUR_LOGITS], batch[SampleBatch.ACTIONS])
        # use my policy to postprocess other's trajectory.
        batches.append(postprocess_ppo_gae(policy, batch))
    return SampleBatch.concat_samples(batches)


class ValueNetworkMixin2(object):
    def __init__(self, config):
        if config["use_gae"]:

            @make_tf_callable(self.get_session(), True)
            def value_batch(ob, prev_action, prev_reward):
                # We do not support recurrent network now.
                model_out, _ = self.model(
                    {
                        SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                        SampleBatch.PREV_ACTIONS: tf.
                        convert_to_tensor(prev_action),
                        SampleBatch.PREV_REWARDS: tf.
                        convert_to_tensor(prev_reward),
                        "is_training": tf.convert_to_tensor(False),
                    }
                )
                return self.model.value_function()
        else:

            @make_tf_callable(self.get_session(), True)
            def value_batch(ob, prev_action, prev_reward):
                return tf.zeros_like(prev_reward)

        self._value_batch = value_batch


def setup_mixins_ceppo(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    if not config['disable']:
        ValueNetworkMixin2.__init__(policy, config)
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
        standardize_fields=["advantages"],
        shuffle_sequences=config["shuffle_sequences"]
    )


def wrap_stats_ceppo(policy, train_batch):
    if policy.config[DIVERSITY_ENCOURAGING]:
        return wrap_stats_fn(policy, train_batch)
    ret = kl_and_loss_stats(policy, train_batch)
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


def loss_ceppo(policy, model, dist_class, train_batch):
    if policy.config[DIVERSITY_ENCOURAGING]:
        return extra_loss_ppo_loss(policy, model, dist_class, train_batch)
    if policy.config[CURIOSITY]:
        # create policy.novelty_loss for adaptively adjust curiosity.
        novelty_loss_mse(policy, model, dist_class, train_batch)
    return ppo_surrogate_loss(policy, model, dist_class, train_batch)


CEPPOTFPolicy = AdaptiveExtraLossPPOTFPolicy.with_updates(
    name="CEPPOTFPolicy",
    get_default_config=lambda: ceppo_default_config,
    postprocess_fn=postprocess_ceppo,
    loss_fn=loss_ceppo,
    before_loss_init=setup_mixins_ceppo,
    stats_fn=wrap_stats_ceppo,
    mixins=mixin_list + [AddLossMixin, NoveltyParamMixin, ValueNetworkMixin2]
)

CEPPOTrainer = AdaptiveExtraLossPPOTrainer.with_updates(
    name="CEPPO",
    after_optimizer_step=wrap_after_train_result_ceppo,
    default_config=ceppo_default_config,
    default_policy=CEPPOTFPolicy,
    validate_config=validate_and_rewrite_config,
    make_policy_optimizer=choose_policy_optimizer_modified
)
