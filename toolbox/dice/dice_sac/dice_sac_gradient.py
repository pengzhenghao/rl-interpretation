from ray.rllib.agents.sac.sac_policy import minimize_and_clip

from toolbox.dice.dice_loss import tf, CLIP_DIVERSITY_GRADIENT, _flatten


def dice_sac_gradient(policy, optimizer, loss):
    if policy.config["grad_norm_clipping"] is not None:
        actor_grads_and_vars = minimize_and_clip(
            optimizer,
            policy.actor_loss,
            var_list=policy.model.policy_variables(),
            clip_val=policy.config["grad_norm_clipping"])
        if policy.config["twin_q"]:
            q_variables = policy.model.q_variables()
            half_cutoff = len(q_variables) // 2
            critic_grads_and_vars = []
            critic_grads_and_vars += minimize_and_clip(
                optimizer,
                policy.critic_loss[0],
                var_list=q_variables[:half_cutoff],
                clip_val=policy.config["grad_norm_clipping"])
            critic_grads_and_vars += minimize_and_clip(
                optimizer,
                policy.critic_loss[1],
                var_list=q_variables[half_cutoff:],
                clip_val=policy.config["grad_norm_clipping"])
        else:
            critic_grads_and_vars = minimize_and_clip(
                optimizer,
                policy.critic_loss[0],
                var_list=policy.model.q_variables(),
                clip_val=policy.config["grad_norm_clipping"])
        alpha_grads_and_vars = minimize_and_clip(
            optimizer,
            policy.alpha_loss,
            var_list=[policy.model.log_alpha],
            clip_val=policy.config["grad_norm_clipping"])
    else:
        actor_grads_and_vars = policy._actor_optimizer.compute_gradients(
            policy.actor_loss, var_list=policy.model.policy_variables())
        if policy.config["twin_q"]:
            q_variables = policy.model.q_variables()
            half_cutoff = len(q_variables) // 2
            base_q_optimizer, twin_q_optimizer = policy._critic_optimizer
            critic_grads_and_vars = base_q_optimizer.compute_gradients(
                policy.critic_loss[0], var_list=q_variables[:half_cutoff]
            ) + twin_q_optimizer.compute_gradients(
                policy.critic_loss[1], var_list=q_variables[half_cutoff:])
        else:
            critic_grads_and_vars = policy._critic_optimizer[
                0].compute_gradients(
                policy.critic_loss[0], var_list=policy.model.q_variables())
        alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(
            policy.alpha_loss, var_list=[policy.model.log_alpha])

    policy_grad = actor_grads_and_vars

    if policy.config["grad_norm_clipping"] is not None:
        diversity_grad = minimize_and_clip(
            optimizer, policy.diversity_loss,
            var_list=policy.model.policy_variables(),
            clip_val=policy.config["grad_norm_clipping"]
        )
    else:
        diversity_grad = policy._actor_optimizer.compute_gradients(
            policy.diversity_loss, var_list=policy.model.policy_variables()
        )

    return_gradients = {}
    policy_grad_flatten = []
    policy_grad_info = []
    diversity_grad_flatten = []
    diversity_grad_info = []

    # First, flatten task gradient and diversity gradient into two vector.
    for (pg, var), (ng, var2) in zip(policy_grad, diversity_grad):
        assert var == var2
        if pg is None:
            return_gradients[var] = (ng, var2)
            continue
        if ng is None:
            return_gradients[var] = (pg, var)
            continue

        pg_flat, pg_shape, pg_flat_shape = _flatten(pg)
        policy_grad_flatten.append(pg_flat)
        policy_grad_info.append((pg_flat_shape, pg_shape, var))

        ng_flat, ng_shape, ng_flat_shape = _flatten(ng)
        diversity_grad_flatten.append(ng_flat)
        diversity_grad_info.append((ng_flat_shape, ng_shape))

    policy_grad_flatten = tf.concat(policy_grad_flatten, 0)
    diversity_grad_flatten = tf.concat(diversity_grad_flatten, 0)

    # Second, compute the norm of two gradient.
    policy_grad_norm = tf.linalg.l2_normalize(policy_grad_flatten)
    diversity_grad_norm = tf.linalg.l2_normalize(diversity_grad_flatten)

    # Third, compute the bisector.
    final_grad = tf.linalg.l2_normalize(policy_grad_norm + diversity_grad_norm)

    # Fourth, compute the length of the final gradient.
    pg_length = tf.norm(tf.multiply(policy_grad_flatten, final_grad))
    ng_length = tf.norm(tf.multiply(diversity_grad_flatten, final_grad))
    if policy.config[CLIP_DIVERSITY_GRADIENT]:
        ng_length = tf.minimum(pg_length, ng_length)
    tg_lenth = (pg_length + ng_length) / 2

    final_grad = final_grad * tg_lenth

    # add some stats.
    policy.gradient_cosine_similarity = tf.reduce_sum(
        tf.multiply(policy_grad_norm, diversity_grad_norm)
    )
    policy.policy_grad_norm = tf.norm(policy_grad_flatten)
    policy.diversity_grad_norm = tf.norm(diversity_grad_flatten)

    # Fifth, split the flatten vector into the original form as the final
    # gradients.
    count = 0
    for idx, (flat_shape, org_shape, var) in enumerate(policy_grad_info):
        assert flat_shape is not None
        size = flat_shape.as_list()[0]
        grad = final_grad[count:count + size]
        return_gradients[var] = (tf.reshape(grad, org_shape), var)
        count += size

    if policy.config["grad_clip"] is not None:
        ret_grads = [return_gradients[var][0] for _, var in policy_grad]
        clipped_grads, _ = tf.clip_by_global_norm(
            ret_grads, policy.config["grad_clip"])
        actor_grads_and_vars = [(g, return_gradients[var][1])
                                for g, (_, var) in
                                zip(clipped_grads, policy_grad)]
    else:
        actor_grads_and_vars = [
            return_gradients[var] for _, var in policy_grad
        ]

    # save these for later use in build_apply_op
    policy._actor_grads_and_vars = [(g, v) for (g, v) in actor_grads_and_vars
                                    if g is not None]
    policy._critic_grads_and_vars = [(g, v) for (g, v) in critic_grads_and_vars
                                     if g is not None]
    policy._alpha_grads_and_vars = [(g, v) for (g, v) in alpha_grads_and_vars
                                    if g is not None]

    grads_and_vars = (
            policy._actor_grads_and_vars + policy._critic_grads_and_vars +
            policy._alpha_grads_and_vars)

    return grads_and_vars
