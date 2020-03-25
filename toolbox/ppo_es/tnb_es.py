from ray import tune
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import setup_mixins, ValueNetworkMixin, \
    KLCoeffMixin, LearningRateSchedule, EntropyCoeffSchedule, SampleBatch
from ray.tune.util import merge_dicts

from toolbox import initialize_ray
from toolbox.distance import get_kl_divergence
from toolbox.ipd.tnb import validate_config as validate_config_TNBTrainer
from toolbox.ipd.tnb_policy import NoveltyValueNetworkMixin, TNBPolicy, \
    tnb_default_config, BEHAVIOUR_LOGITS
from toolbox.ipd.tnb_utils import *
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.modified_rllib.multi_gpu_optimizer import \
    LocalMultiGPUOptimizerCorrectedNumberOfSampled
from toolbox.dies.ppo_es import PPOESTrainer, \
    validate_config as validate_config_PPOESTrainer

tnbes_config = merge_dicts(
    merge_dicts(DEFAULT_CONFIG, tnb_default_config),
    dict(
        update_steps=100000,
        use_tnb_plus=False,
        novelty_type="mse",  # must in ['mse', 'kl']
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


class ComputeNoveltyMixin(object):
    def __init__(self):
        self.enable_novelty = True

    def compute_novelty(self, my_batch, others_batches, episode):
        if not others_batches:
            return np.zeros_like(
                my_batch[SampleBatch.REWARDS], dtype=np.float32
            )

        replays = []
        for (other_policy, _) in others_batches.values():
            _, _, info = other_policy.compute_actions(
                my_batch[SampleBatch.CUR_OBS]
            )
            replays.append(info[BEHAVIOUR_LOGITS])

        assert replays

        if self.config["novelty_type"] == "kl":
            return np.mean(
                [
                    get_kl_divergence(
                        my_batch[BEHAVIOUR_LOGITS], logit, mean=False
                    ) for logit in replays
                ],
                axis=0
            )

        elif self.config["novelty_type"] == "mse":
            replays = [np.split(logit, 2, axis=1)[0] for logit in replays]
            my_act = np.split(my_batch[BEHAVIOUR_LOGITS], 2, axis=1)[0]
            return np.mean(
                [
                    (np.square(my_act - other_act)).mean(1)
                    for other_act in replays
                ],
                axis=0
            )
        else:
            raise NotImplementedError()


def setup_mixins_tnb(policy, action_space, obs_space, config):
    setup_mixins(policy, action_space, obs_space, config)
    NoveltyValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    ComputeNoveltyMixin.__init__(policy)


TNBESPolicy = TNBPolicy.with_updates(
    name="TNBESPolicy",
    get_default_config=lambda: tnbes_config,
    before_loss_init=setup_mixins_tnb,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, NoveltyValueNetworkMixin, ComputeNoveltyMixin
    ]
)


def validate_config(config):
    validate_config_PPOESTrainer(config)
    validate_config_TNBTrainer(config)
    assert config['novelty_type'] in ['mse', 'kl']


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
    initialize_ray(test_mode=True, local_mode=False)
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
        "update_steps": 1000,
        "use_tnb_plus": tune.grid_search([True, False]),
        "novelty_type": tune.grid_search(["mse", 'kl']),
        "use_novelty_value_network": True
    }
    tune.run(
        TNBESTrainer,
        name="DELETEME_TEST",
        verbose=2,
        stop={"timesteps_total": 10000},
        config=config
    )
