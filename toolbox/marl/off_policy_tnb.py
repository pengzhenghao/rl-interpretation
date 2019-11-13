"""This file implement the modified version of TNB."""
import tensorflow as tf

from toolbox.marl.extra_loss_ppo_trainer import novelty_loss, \
    ppo_surrogate_loss, ExtraLossPPOTFPolicy, DEFAULT_CONFIG, merge_dicts, \
    ExtraLossPPOTrainer, validate_config_basic, \
    kl_and_loss_stats_without_total_loss

tnb_ppo_default_config = merge_dicts(
    DEFAULT_CONFIG, dict(joint_dataset_sample_batch_size=200)
)


def tnb_loss(policy, model, dist_class, train_batch):
    """Add novelty loss with original ppo loss using TNB method"""
    original_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    nov_loss = novelty_loss(policy, model, dist_class, train_batch)
    # In rllib convention, loss_fn should return one single tensor
    # however, there is no explicit bugs happen returning a list.
    return [original_loss, nov_loss]


def _flatten(tensor):
    flat = tf.reshape(tensor, shape=[-1])
    return flat, tensor.shape, flat.shape


def tnb_gradients(policy, optimizer, loss):
    policy_grad = optimizer.compute_gradients(loss[0])
    novelty_grad = optimizer.compute_gradients(loss[1])

    # flatten the policy_grad
    policy_grad_flatten = []
    policy_grad_info = []
    for idx, (pg, var) in enumerate(policy_grad):
        if novelty_grad[idx][0] is None:
            # Some variables do not related to novelty, so the grad is None
            policy_grad_info.append((None, None, var))
            continue
        pg_flat, pg_shape, pg_flat_shape = _flatten(pg)
        policy_grad_flatten.append(pg_flat)
        policy_grad_info.append((pg_flat_shape, pg_shape, var))
    policy_grad_flatten = tf.concat(policy_grad_flatten, 0)

    # flatten the novelty grad
    novelty_grad_flatten = []
    novelty_grad_info = []
    for ng, _ in novelty_grad:
        if ng is None:
            novelty_grad_info.append((None, None))
            continue
        pg_flat, pg_shape, pg_flat_shape = _flatten(ng)
        novelty_grad_flatten.append(pg_flat)
        novelty_grad_info.append((pg_flat_shape, pg_shape))
    novelty_grad_flatten = tf.concat(novelty_grad_flatten, 0)

    # implement the logic of TNB
    policy_grad_norm = tf.linalg.l2_normalize(policy_grad_flatten)
    novelty_grad_norm = tf.linalg.l2_normalize(novelty_grad_flatten)
    cos_similarity = tf.reduce_sum(
        tf.multiply(policy_grad_norm, novelty_grad_norm)
    )

    def less_90_deg():
        tg = policy_grad_norm + novelty_grad_norm
        tg = tf.linalg.l2_normalize(tg)
        mag = (
            tf.norm(tf.multiply(policy_grad_flatten, tg)) +
            tf.norm(tf.multiply(novelty_grad_flatten, tg))
        ) / 2
        tg = tg * mag
        return tg

    def greater_90_deg():
        tg = -cos_similarity * novelty_grad_norm + policy_grad_norm
        tg = tf.linalg.l2_normalize(tg)
        tg = tg * tf.norm(tf.multiply(policy_grad_norm, tg))
        # Here is a modification to the origianl TNB, we add 1/2 here.
        tg = tg / 2
        return tg

    total_grad = tf.cond(cos_similarity > 0, less_90_deg, greater_90_deg)

    # reshape back the gradients
    return_gradients = []
    count = 0
    for idx, (flat_shape, org_shape, var) in enumerate(policy_grad_info):
        if flat_shape is None:
            return_gradients.append((None, var))
            continue
        size = flat_shape.as_list()[0]
        grad = total_grad[count:count + size]
        return_gradients.append((tf.reshape(grad, org_shape), var))
        count += size

    return return_gradients


TNBPPOTFPolicy = ExtraLossPPOTFPolicy.with_updates(
    name="TNBPPOTFPolicy",
    get_default_config=lambda: tnb_ppo_default_config,
    loss_fn=tnb_loss,
    gradients_fn=tnb_gradients,
    stats_fn=kl_and_loss_stats_without_total_loss
)

TNBPPOTrainer = ExtraLossPPOTrainer.with_updates(
    name="TNBPPO",
    default_config=tnb_ppo_default_config,
    validate_config=validate_config_basic,
    default_policy=TNBPPOTFPolicy
)

if __name__ == '__main__':
    from toolbox import initialize_ray
    from ray import tune
    from toolbox.utils import get_local_dir
    from toolbox.train.train_individual_marl import on_train_result
    from toolbox.marl.multiagent_env_wrapper import MultiAgentEnvWrapper

    num_agents = 5
    num_gpus = 0

    # This is only test code.
    initialize_ray(test_mode=True, local_mode=False, num_gpus=num_gpus)

    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]

    env_config = {"env_name": "BipedalWalker-v2", "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "num_gpus": num_gpus,
        "log_level": "DEBUG",
        "joint_dataset_sample_batch_size": 131,
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

    tune.run(
        TNBPPOTrainer,
        local_dir=get_local_dir(),
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        stop={"timesteps_total": 2000},
        config=config,
    )
