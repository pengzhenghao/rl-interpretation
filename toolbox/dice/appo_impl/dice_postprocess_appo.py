"""
Implement the Collaborative Exploration here.

In postprocess_dice, which is called for each policy with input other
policies' batches, we fuse the batches collected by this policy and other
policies. We also compute the diversity reward and diversity advantage of this
policy.
"""
from ray.rllib.agents.impala.vtrace_policy import BEHAVIOUR_LOGITS
# from ray.rllib.agents.ppo.appo_policy import \
#     postprocess_trajectory as original_postprocess
from ray.rllib.agents.ppo.ppo_policy import ACTION_LOGP, \
    BEHAVIOUR_LOGITS
from ray.rllib.evaluation.postprocessing import discount, compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch

from toolbox.dice.utils import *

MY_LOGIT = "my_logits"


def original_postprocess(policy,
                         sample_batch,
                         other_agent_batches=None,
                         episode=None):
    if not policy.config["vtrace"]:
        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            next_state = []
            for i in range(policy.num_state_tensors()):
                next_state.append([sample_batch["state_out_{}".format(i)][-1]])
            last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                                   sample_batch[SampleBatch.ACTIONS][-1],
                                   sample_batch[SampleBatch.REWARDS][-1],
                                   *next_state)
        batch = compute_advantages(
            sample_batch,
            last_r,
            policy.config["gamma"],
            policy.config["lambda"],
            use_gae=policy.config["use_gae"])
    else:
        batch = sample_batch
    # del batch.data["new_obs"]  # not used, so save some bandwidth
    return batch


# TODO this may be problematic. I don't see clearly how things going
def postprocess_dice(policy, sample_batch, others_batches, episode):
    if not policy.loss_initialized():
        batch = original_postprocess(policy, sample_batch)
        batch[DIVERSITY_REWARDS] = batch["advantages"].copy()
        batch[DIVERSITY_VALUE_TARGETS] = batch["advantages"].copy()
        batch[DIVERSITY_ADVANTAGES] = batch["advantages"].copy()
        batch['other_action_logp'] = batch[ACTION_LOGP].copy()
        return batch

    if (not policy.config[PURE_OFF_POLICY]) or (not others_batches):
        batch = sample_batch.copy()
        batch = original_postprocess(policy, batch)
        batch[MY_LOGIT] = batch[BEHAVIOUR_LOGITS]
        batch = postprocess_diversity(policy, batch, others_batches)
        batches = [batch]
    else:
        batches = []

    for pid, (other_policy, other_batch_raw) in others_batches.items():

        ##########
        # This processing is not acceptable. we should remove multiple policies
        # in system. and use different worker latest agent to serve as a
        # population.
        ##########

        # other_batch_raw is the data collected by other polices.
        if policy.config[ONLY_TNB]:
            break
        if other_batch_raw is None:
            continue
        other_batch_raw = other_batch_raw.copy()

        # Replay this policy to get the action distribution of this policy.
        replay_result = policy.compute_actions(
            other_batch_raw[SampleBatch.CUR_OBS]
        )[2]
        other_batch_raw[MY_LOGIT] = replay_result[BEHAVIOUR_LOGITS]

        # Compute the diversity reward and diversity advantage of this batch.
        other_batch_raw = postprocess_diversity(
            policy, other_batch_raw, others_batches
        )

        # Compute the task advantage of this batch.
        batches.append(original_postprocess(policy, other_batch_raw))

    # Merge all batches.
    batch = SampleBatch.concat_samples(batches) if len(batches) != 1 \
        else batches[0]

    # TODO we should remove this but this cause error
    del batch.data['new_obs']  # save memory
    del batch.data['action_prob']
    return batch


def postprocess_diversity(policy, batch, others_batches):
    """Compute the diversity for this policy against other policies using this
    batch."""

    # do not compute the diversity in each worker. instead, compute the
    # diversity in the aggregation stage (real optimization stage)
    batch[DIVERSITY_ADVANTAGES] = np.ones_like(batch["rewards"])
    batch[DIVERSITY_VALUE_TARGETS] = np.ones_like(batch["rewards"])
    batch[DIVERSITY_REWARDS] = np.ones_like(batch["rewards"])
    return batch



    # Compute diversity and add a new entry of batch: diversity_reward
    batch[DIVERSITY_REWARDS] = policy.compute_diversity(batch, others_batches)

    # Compute the diversity advantage. We mock the computing of task advantage
    # but simply replace the task reward with the diversity reward.
    completed = batch["dones"][-1]
    if completed:
        last_r_diversity = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([batch["state_out_{}".format(i)][-1]])
        last_r_diversity = policy._diversity_value(
            batch[SampleBatch.NEXT_OBS][-1], batch[SampleBatch.ACTIONS][-1],
            batch[DIVERSITY_REWARDS][-1], *next_state
        )
    diversity_advantages, diversity_value_target = \
        _compute_advantages_for_diversity(
            rewards=batch[DIVERSITY_REWARDS],
            last_r=last_r_diversity,
            gamma=policy.config["gamma"],
            lambda_=policy.config["lambda"],
            values=batch[DIVERSITY_VALUES]
            if policy.config[USE_DIVERSITY_VALUE_NETWORK] else None,
            use_gae=policy.config[USE_DIVERSITY_VALUE_NETWORK]
        )
    batch[DIVERSITY_ADVANTAGES] = diversity_advantages
    batch[DIVERSITY_VALUE_TARGETS] = diversity_value_target
    return batch


def _compute_advantages_for_diversity(
        rewards, last_r, gamma, lambda_, values, use_gae=True
):
    """Compute the diversity advantage."""
    if use_gae:
        vpred_t = np.concatenate([values, np.array([last_r])])
        delta_t = (rewards + gamma * vpred_t[1:] - vpred_t[:-1])
        advantage = discount(delta_t, gamma * lambda_)
        value_target = (advantage + values).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate([rewards, np.array([last_r])])
        advantage = discount(rewards_plus_v, gamma)[:-1]
        value_target = np.zeros_like(advantage, dtype=np.float32)
    advantage = advantage.copy().astype(np.float32)
    return advantage, value_target
