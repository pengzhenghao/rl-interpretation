from ray.tune.utils import merge_dicts

from toolbox import initialize_ray, train
from toolbox.dice import DiCETrainer, utils as dice_utils
from toolbox.dice.utils import dice_default_config
from toolbox.dies.es_utils import run_evolution_strategies
from toolbox.marl import get_marl_env_config

dies_default_config = merge_dicts(
    dice_default_config,
    {
        "update_steps": 100000,
        # callbacks={"on_train_result": on_train_result}  # already there
        dice_utils.DELAY_UPDATE: False,
        dice_utils.TWO_SIDE_CLIP_LOSS: False,
        dice_utils.ONLY_TNB: True,
        dice_utils.NORMALIZE_ADVANTAGE: True,  # May be need to set false
    }
)

DiESTrainer = DiCETrainer.with_updates(
    name="DiES",
    default_config=dies_default_config,
    after_train_result=run_evolution_strategies
)

if __name__ == '__main__':
    env_name = "CartPole-v0"
    num_agents = 3
    config = {
        "num_sgd_iter": 2,
        "train_batch_size": 400,
        "update_steps": 1000,
        **get_marl_env_config(env_name, num_agents)
    }
    initialize_ray(test_mode=True, local_mode=True)
    train(
        DiESTrainer,
        config,
        exp_name="DELETE_ME_TEST",
        stop={"timesteps_total": 10000},
        test_mode=True
    )
