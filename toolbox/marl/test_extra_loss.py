from ray import tune

from toolbox import initialize_ray, get_local_dir
from toolbox.marl import MultiAgentEnvWrapper, on_train_result
from toolbox.marl.adaptive_extra_loss import AdaptiveExtraLossPPOTrainer
from toolbox.marl.adaptive_tnb import AdaptiveTNBPPOTrainer
from toolbox.marl.extra_loss_ppo_trainer import ExtraLossPPOTrainer
from toolbox.marl.smart_adaptive_extra_loss import \
    SmartAdaptiveExtraLossPPOTrainer
from toolbox.marl.task_novelty_bisector import TNBPPOTrainer


# test_default_config = {}
def _get_default_test_config(num_agents, env_name, num_gpus):
    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]
    env_config = {"env_name": env_name, "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "num_gpus": num_gpus,
        "log_level": "DEBUG",
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
    }
    return config


def _base(
        trainer,
        local_mode=False,
        extra_config=None,
        t=500,
        env_name="BipedalWalker-v2",
        num_agents=3
):
    # num_agents = 3
    num_gpus = 0

    initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=num_gpus)

    config = _get_default_test_config(num_agents, env_name, num_gpus)
    if extra_config:
        config.update(extra_config)

    return tune.run(
        trainer,
        local_dir=get_local_dir(),
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        stop={"timesteps_total": t},
        config=config
    )


def _base_marl(
        trainer,
        local_mode=False,
        extra_config=None,
        t=500,
        env_name="BipedalWalker-v2"
):
    config = {
        "joint_dataset_sample_batch_size": 30,
        "callbacks": {
            "on_train_result": on_train_result
        }
    }
    if extra_config:
        config.update(extra_config)
    _base(trainer, local_mode, config, t, env_name)


def test_adaptive_extra_loss_trainer1():
    _base_marl(
        AdaptiveExtraLossPPOTrainer,
        local_mode=False,
        extra_config={"use_joint_dataset": False}
    )


def test_adaptive_extra_loss_trainer2():
    _base_marl(
        AdaptiveExtraLossPPOTrainer,
        local_mode=False,
        extra_config={"use_joint_dataset": True}
    )


def test_extra_loss_ppo_trainer1(local_mode=False):
    _base_marl(
        ExtraLossPPOTrainer,
        local_mode=local_mode,
        extra_config={"use_joint_dataset": False}
    )


def test_extra_loss_ppo_trainer2():
    _base_marl(
        ExtraLossPPOTrainer,
        local_mode=False,
        extra_config={"use_joint_dataset": True}
    )


def test_smart_adaptive_extra_loss_trainer1(local_mode=False):
    _base_marl(
        SmartAdaptiveExtraLossPPOTrainer,
        local_mode, {
            "waiting_iteration": 2,
            "use_joint_dataset": True
        },
        t=10000
    )


def test_smart_adaptive_extra_loss_trainer2(local_mode=False):
    _base_marl(
        SmartAdaptiveExtraLossPPOTrainer,
        local_mode, {
            "waiting_iteration": 2,
            "use_joint_dataset": False
        },
        t=10000
    )


def test_smart_adaptive_extra_loss_trainer3(local_mode=False):
    _base_marl(
        SmartAdaptiveExtraLossPPOTrainer,
        local_mode,
        {
            "waiting_iteration": 2,
            "use_joint_dataset": True,
            "performance_evaluation_metric": "mean"
        },
    )


def test_smart_adaptive_extra_loss_trainer4(local_mode=False):
    _base_marl(
        SmartAdaptiveExtraLossPPOTrainer,
        local_mode=local_mode,
        env="HumanoidBulletEnv-v0"
    )


def test_smart_adaptive_extra_loss_trainer5(local_mode=False):
    _base_marl(
        SmartAdaptiveExtraLossPPOTrainer,
        local_mode=local_mode,
        env="CartPole-v0"
    )


def test_adaptive_tnb():
    _base_marl(AdaptiveTNBPPOTrainer, extra_config={})
    _base_marl(
        AdaptiveTNBPPOTrainer, extra_config={"clip_novelty_gradient": False}
    )
    _base_marl(
        AdaptiveTNBPPOTrainer, extra_config={"use_second_component": True}
    )


def test_tnb_ppo_trainer(use_joint_dataset=True, local_mode=False):
    _base_marl(
        TNBPPOTrainer,
        local_mode=local_mode,
        extra_config={"use_joint_dataset": use_joint_dataset}
    )


def test_restore():
    from toolbox.evaluate import restore_agent
    initialize_ray()
    ckpt = "~/ray_results/1114-tnb_4in1" \
           "/TNBPPO_MultiAgentEnvWrapper_2_novelty_mode=min," \
           "use_joint_dataset=False_2019-11-14_10-30-29l456mu0o" \
           "/checkpoint_60/checkpoint-60"
    marl_agent = restore_agent(TNBPPOTrainer, ckpt, MultiAgentEnvWrapper)


if __name__ == '__main__':
    # test_restore()
    test_smart_adaptive_extra_loss_trainer4(True)
    # test_smart_adaptive_extra_loss_trainer5(True)
