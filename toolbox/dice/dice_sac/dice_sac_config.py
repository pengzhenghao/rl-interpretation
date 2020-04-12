from ray.rllib.agents.sac.sac import DEFAULT_CONFIG as sac_default_config

import toolbox.dice.utils as constants
from toolbox.utils import merge_dicts

dice_sac_default_config = merge_dicts(
    sac_default_config, {

        "clip_param": 0.3,

        "grad_clip": 40.0,

        constants.USE_BISECTOR: True,
        constants.USE_DIVERSITY_VALUE_NETWORK: False,
        constants.DELAY_UPDATE: True,
        constants.TWO_SIDE_CLIP_LOSS: True,
        constants.ONLY_TNB: False,
        constants.NORMALIZE_ADVANTAGE: False,
        constants.CLIP_DIVERSITY_GRADIENT: True,
        constants.DIVERSITY_REWARD_TYPE: "mse",
        constants.PURE_OFF_POLICY: False,
        # "tau": 5e-3,  # <<== SAC already have this
        "callbacks": {
            "on_train_result": constants.on_train_result,
            "on_postprocess_traj": constants.on_postprocess_traj
        }}
)
