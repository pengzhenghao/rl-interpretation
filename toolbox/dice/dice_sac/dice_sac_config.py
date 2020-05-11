from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.sac.sac import DEFAULT_CONFIG as sac_default_config

import toolbox.dice.utils as constants
from toolbox.utils import merge_dicts


class DiCESACCallbacks(DefaultCallbacks):
    def on_postprocess_trajectory(self, *args, **kwargs):
        constants.on_postprocess_trajectory(*args, **kwargs)


dice_sac_default_config = merge_dicts(
    sac_default_config,
    {

        # PPO loss for diversity
        # "clip_param": 0.3,
        # "lambda": 1.0,
        "grad_clip": 40.0,

        # "rollout_fragment_length": 50,
        constants.USE_BISECTOR: True,
        constants.USE_DIVERSITY_VALUE_NETWORK: False,
        constants.DELAY_UPDATE: True,
        # constants.TWO_SIDE_CLIP_LOSS: True,
        constants.ONLY_TNB: False,
        constants.NORMALIZE_ADVANTAGE: False,
        constants.CLIP_DIVERSITY_GRADIENT: True,
        constants.DIVERSITY_REWARD_TYPE: "mse",
        constants.PURE_OFF_POLICY: False,
        "normalize_actions": False,
        "env_config": {
            "normalize_actions": False
        },

        # "tau": 5e-3,  # <<== SAC already have this
        "callbacks": DiCESACCallbacks
    }
)
