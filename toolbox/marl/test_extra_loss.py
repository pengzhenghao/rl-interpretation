from ray import tune

from toolbox import initialize_ray, get_local_dir
from toolbox.marl import MultiAgentEnvWrapper, on_train_result
from toolbox.marl.adaptive_extra_loss import AdaptiveExtraLossPPOTrainer
from toolbox.marl.adaptive_tnb import AdaptiveTNBPPOTrainer
from toolbox.marl.extra_loss_ppo_trainer import ExtraLossPPOTrainer
from toolbox.marl.smart_adaptive_extra_loss import \
    SmartAdaptiveExtraLossPPOTrainer
from toolbox.marl.task_novelty_bisector import TNBPPOTrainer


def _base(trainer, local_mode=False, extra_config=None, t=500):
    num_agents = 3
    num_gpus = 0

    # This is only test code.
    initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=num_gpus)

    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]

    env_config = {"env_name": "BipedalWalker-v2", "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "num_gpus": num_gpus,
        "log_level": "DEBUG",
        "joint_dataset_sample_batch_size": 30,
        "sample_batch_size": 20,
        "train_batch_size": 40,
        "sgd_minibatch_size": 16,
        "multiagent": {
            "policies": {
                i: (None, env.observation_space, env.action_space, {})
                for i in policy_names
            },
            "policy_mapping_fn": lambda x: x,
        },
        "callbacks": {
            "on_train_result": on_train_result
        },
    }
    if extra_config:
        config.update(extra_config)

    tune.run(
        trainer,
        local_dir=get_local_dir(),
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        stop={"timesteps_total": t},
        config=config
    )


def test_adaptive_extra_loss_trainer1():
    _base(
        AdaptiveExtraLossPPOTrainer,
        local_mode=False,
        extra_config={"use_joint_dataset": False}
    )


def test_adaptive_extra_loss_trainer2():
    _base(
        AdaptiveExtraLossPPOTrainer,
        local_mode=False,
        extra_config={"use_joint_dataset": True}
    )


def test_extra_loss_ppo_trainer1():
    _base(
        ExtraLossPPOTrainer,
        local_mode=False,
        extra_config={"use_joint_dataset": False}
    )


def test_extra_loss_ppo_trainer2():
    _base(
        ExtraLossPPOTrainer,
        local_mode=False,
        extra_config={"use_joint_dataset": True}
    )


def test_smart_adaptive_extra_loss_trainer1(local_mode=False):
    _base(
        SmartAdaptiveExtraLossPPOTrainer,
        local_mode, {
            "waiting_iteration": 2,
            "use_joint_dataset": True
        },
        t=10000
    )


def test_smart_adaptive_extra_loss_trainer2(local_mode=False):
    _base(
        SmartAdaptiveExtraLossPPOTrainer,
        local_mode, {
            "waiting_iteration": 2,
            "use_joint_dataset": False
        },
        t=10000
    )


def test_smart_adaptive_extra_loss_trainer3(local_mode=False):
    _base(
        SmartAdaptiveExtraLossPPOTrainer,
        local_mode,
        {
            "waiting_iteration": 2,
            "use_joint_dataset": True,
            "performance_evaluation_metric": "mean"
        },
    )


def test_adaptive_tnb():
    _base(AdaptiveTNBPPOTrainer, extra_config={})
    _base(AdaptiveTNBPPOTrainer, extra_config={"clip_novelty_gradient": False})
    _base(AdaptiveTNBPPOTrainer, extra_config={"use_second_component": True})


def test_tnb_ppo_trainer(use_joint_dataset=True, local_mode=False):
    _base(
        TNBPPOTrainer,
        local_mode=local_mode,
        extra_config={"use_joint_dataset": use_joint_dataset}
    )
