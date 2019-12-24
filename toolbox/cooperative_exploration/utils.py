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