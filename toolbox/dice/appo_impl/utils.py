from ray.rllib.agents.ppo.appo import DEFAULT_CONFIG as APPO_DEFAULT

from toolbox.dice.utils import *

dice_appo_default_config = merge_dicts(
    APPO_DEFAULT, {
        USE_BISECTOR: True,
        USE_DIVERSITY_VALUE_NETWORK: False,
        DELAY_UPDATE: True,
        TWO_SIDE_CLIP_LOSS: True,
        ONLY_TNB: False,
        NORMALIZE_ADVANTAGE: False,
        CLIP_DIVERSITY_GRADIENT: True,
        DIVERSITY_REWARD_TYPE: "mse",
        PURE_OFF_POLICY: False,
        "tau": 5e-3,
        "callbacks": {
            "on_train_result": on_train_result,
            "on_postprocess_traj": on_postprocess_traj
        }
    }
)
