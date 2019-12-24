import logging

from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, PPOTrainer, PPOTFPolicy
from ray.rllib.agents.ppo.ppo_policy import setup_mixins, ValueNetworkMixin, \
    KLCoeffMixin, LearningRateSchedule, EntropyCoeffSchedule
from ray.tune.util import merge_dicts

logger = logging.getLogger(__name__)

ipd_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
    )
)


class AgentPoolMixim(object):

    def __init__(self):
        pass


def setup_mixins_ipd(policy, action_space, obs_space, config):
    setup_mixins(policy, action_space, obs_space, config)
    AgentPoolMixim.__init__(policy)


IPDPolicy = PPOTFPolicy.with_updates(
    name="IPDPolicy",
    get_default_config=lambda: ipd_default_config,
    before_loss_init=setup_mixins,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
            ValueNetworkMixin, AgentPoolMixim]
)

IPDTrainer = PPOTrainer.with_updates(
    name="IPD",
    default_config=ipd_default_config,
    default_policy=IPDPolicy
)
