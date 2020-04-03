from ray.rllib.agents.ppo.appo import DEFAULT_CONFIG as APPO_DEFAULT

from toolbox.dice.utils import *

I_AM_CLONE = "_i_am_clone"

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

        "num_agents": 1,  # Control the agent population size

        "tau": 5e-3,
        "callbacks": {
            # TODO(checked) since we do not maintain a policies map anymore,
            #  so hard for
            #  us to compute the distance between agents.
            #  think about whether we need this or how we make this
            # "on_train_result": on_train_result,

            # TODO(checked) remove this counting. please check whether this
            #  would harm
            #  the counting
            # "on_postprocess_traj": on_postprocess_traj
        }
    }
)
