from ray import tune
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import setup_mixins, ValueNetworkMixin, \
    KLCoeffMixin, LearningRateSchedule, EntropyCoeffSchedule, SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.tune.util import merge_dicts

from toolbox import initialize_ray
from toolbox.ipd.tnb import validate_config as validate_config_TNBTrainer
from toolbox.ipd.tnb_policy import NoveltyValueNetworkMixin, TNBPolicy, \
    tnb_default_config, compute_advantages
from toolbox.ipd.tnb_utils import *
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.modified_rllib.multi_gpu_optimizer import \
    LocalMultiGPUOptimizerCorrectedNumberOfSampled
from toolbox.ppo_es.ppo_es import PPOESTrainer, \
    validate_config as validate_config_PPOESTrainer

tnbes_config = merge_dicts(
    merge_dicts(DEFAULT_CONFIG, tnb_default_config), dict(
        update_steps=100000,
        novelty_threshold=0.5,
        use_tnb_plus=False,
        use_novelty_value_network=False
    )
)

"""
The main different between the TNBTrainer at toolbox.ipd and here is that
the weight swapping operation is done by passing checkpoint_dict in config
at TNBTrainer. But we do the weight sharing between policies inplace 
immediately after train iteration.

TNBESPolicy remove the AgentPoolMixin.

TNBESTrainer merge the TNBTrainer and PPOESTrainer.
"""


def setup_mixins_tnb(policy, action_space, obs_space, config):
    setup_mixins(policy, action_space, obs_space, config)
    NoveltyValueNetworkMixin.__init__(policy, obs_space, action_space, config)


def postprocess_tnbes(policy, sample_batch, other_batches, episode):
    completed = sample_batch["dones"][-1]
    sample_batch[NOVELTY_REWARDS] = policy.compute_novelty(
        sample_batch[SampleBatch.CUR_OBS], sample_batch[SampleBatch.ACTIONS]
    )

    if completed:
        last_r_novelty = last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(
            sample_batch[SampleBatch.NEXT_OBS][-1],
            sample_batch[SampleBatch.ACTIONS][-1],
            sample_batch[SampleBatch.REWARDS][-1], *next_state
        )
        last_r_novelty = policy._novelty_value(
            sample_batch[SampleBatch.NEXT_OBS][-1],
            sample_batch[SampleBatch.ACTIONS][-1],
            sample_batch[NOVELTY_REWARDS][-1], *next_state
        )

    # compute the advantages of original rewards
    advantages, value_target = compute_advantages(
        sample_batch[SampleBatch.REWARDS],
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        sample_batch[SampleBatch.VF_PREDS],
        use_gae=policy.config["use_gae"]
    )
    sample_batch[Postprocessing.ADVANTAGES] = advantages
    sample_batch[Postprocessing.VALUE_TARGETS] = value_target

    # compute the advantages of novelty rewards
    novelty_advantages, novelty_value_target = compute_advantages(
        rewards=sample_batch[NOVELTY_REWARDS],
        last_r=last_r_novelty,
        gamma=policy.config["gamma"],
        lambda_=policy.config["lambda"],
        values=sample_batch[NOVELTY_VALUES]
        if policy.config['use_novelty_value_network'] else None,
        use_gae=policy.config['use_novelty_value_network']
    )
    sample_batch[NOVELTY_ADVANTAGES] = novelty_advantages
    sample_batch[NOVELTY_VALUE_TARGETS] = novelty_value_target

    return sample_batch


TNBESPolicy = TNBPolicy.with_updates(
    name="TNBESPolicy",
    get_default_config=lambda: tnbes_config,
    before_loss_init=setup_mixins_tnb,
    postprocess_fn=postprocess_tnbes,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, NoveltyValueNetworkMixin
    ]
)


def validate_config(config):
    validate_config_PPOESTrainer(config)
    validate_config_TNBTrainer(config)


def make_policy_optimizer_tnbes(workers, config):
    """The original optimizer has wrong number of trained samples stats.
    So we make little modification and use the corrected optimizer.
    This function is only made for PPO.
    """
    if config["simple_optimizer"]:
        raise NotImplementedError()

    return LocalMultiGPUOptimizerCorrectedNumberOfSampled(
        workers,
        compute_num_steps_sampled=None,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=["advantages", NOVELTY_ADVANTAGES],  # HERE!
        shuffle_sequences=config["shuffle_sequences"]
    )


TNBESTrainer = PPOESTrainer.with_updates(
    name="TNBES",
    validate_config=validate_config,
    make_policy_optimizer=make_policy_optimizer_tnbes,
    default_config=tnbes_config,
    default_policy=TNBESPolicy
)

if __name__ == '__main__':
    # Test codes
    initialize_ray(test_mode=True, local_mode=True)
    env_name = "CartPole-v0"
    num_agents = 3

    config = {
        "num_sgd_iter": 2,
        "train_batch_size": 400,
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": env_name,
            "num_agents": num_agents
        },
        "update_steps": 1000
    }
    tune.run(
        TNBESTrainer,
        name="DELETEME_TEST",
        verbose=2,
        stop={"timesteps_total": 10000},
        config=config
    )
