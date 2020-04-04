from ray.rllib.agents.ppo.appo import DEFAULT_CONFIG as APPO_DEFAULT

from toolbox.utils import merge_dicts

I_AM_CLONE = "_i_am_clone"
USE_BISECTOR = "use_bisector"  # If false, the the DR is disabled.
USE_DIVERSITY_VALUE_NETWORK = "use_diversity_value_network"
DELAY_UPDATE = "delay_update"
CLIP_DIVERSITY_GRADIENT = "clip_diversity_gradient"
DIVERSITY_REWARD_TYPE = "diversity_reward_type"
DIVERSITY_REWARDS = "diversity_rewards"
DIVERSITY_VALUES = "diversity_values"
DIVERSITY_ADVANTAGES = "diversity_advantages"
DIVERSITY_VALUE_TARGETS = "diversity_value_targets"
NORMALIZE_ADVANTAGE = "normalize_advantage"

dice_appo_default_config = merge_dicts(
    APPO_DEFAULT, {
        USE_BISECTOR: True,
        USE_DIVERSITY_VALUE_NETWORK: False,
        DELAY_UPDATE: True,
        NORMALIZE_ADVANTAGE: False,
        CLIP_DIVERSITY_GRADIENT: True,
        DIVERSITY_REWARD_TYPE: "mse",

        "num_agents": 1,  # Control the agent population size
        "tau": 5e-3,
        "clip_param": 0.3,
        "lr": 1e-4,
        "num_sgd_iter": 1,  # In PPO this is 10
        "max_sample_requests_in_flight_per_worker": 1,  # originally 2
        "replay_buffer_num_slots": 0  # disable replay
    }
)
