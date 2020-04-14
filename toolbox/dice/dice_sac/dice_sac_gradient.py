import numpy as np
from ray.rllib.agents.sac.sac_policy import get_dist_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_ops import minimize_and_clip

from toolbox.dice.dice_loss import tf, CLIP_DIVERSITY_GRADIENT, _flatten


def dice_sac_loss(policy, model, _, train_batch):
    model_out_t, _ = model({
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    model_out_tp1, _ = model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    target_model_out_tp1, _ = policy.target_model({
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": policy._get_is_training_placeholder(),
    }, [], None)

    # Discrete case.
    if model.discrete:
        raise NotImplementedError()
        # Get all action probs directly from pi and form their logp.
        log_pis_t = tf.nn.log_softmax(model.get_policy_output(model_out_t), -1)
        policy_t = tf.exp(log_pis_t)
        log_pis_tp1 = tf.nn.log_softmax(
            model.get_policy_output(model_out_tp1), -1)
        policy_tp1 = tf.exp(log_pis_tp1)
        # Q-values.
        q_t = model.get_q_values(model_out_t)
        # Target Q-values.
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(model_out_t)
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1)
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
        q_tp1 -= model.alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = tf.one_hot(
            train_batch[SampleBatch.ACTIONS], depth=q_t.shape.as_list()[-1])
        q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
        q_tp1_best_masked = \
            (1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * \
            q_tp1_best
    # Continuous actions case.
    else:
        # Sample simgle actions from distribution.
        action_dist_class = get_dist_class(policy.config, policy.action_space)
        action_dist_t = action_dist_class(
            model.get_policy_output(model_out_t), policy.model)
        policy_t = action_dist_t.sample()
        log_pis_t = tf.expand_dims(action_dist_t.sampled_action_logp(), -1)
        action_dist_tp1 = action_dist_class(
            model.get_policy_output(model_out_tp1), policy.model)
        policy_tp1 = action_dist_tp1.sample()
        log_pis_tp1 = tf.expand_dims(action_dist_tp1.sampled_action_logp(), -1)

        # Q-values for the actually selected actions.
        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        diversity_q_t = model.get_diversity_q_values(
            model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS])

        # Q-values for current policy in given current state.
        q_t_det_policy = model.get_q_values(model_out_t, policy_t)
        diversity_q_t_det_policy = model.get_diversity_q_values(
            model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy = model.get_twin_q_values(
                model_out_t, policy_t)
            q_t_det_policy = tf.reduce_min(
                (q_t_det_policy, twin_q_t_det_policy), axis=0)

        # target q network evaluation
        q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
                                                 policy_tp1)
        diversity_q_tp1 = policy.target_model.get_diversity_q_values(
            target_model_out_tp1, policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1)

        q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
        diversity_q_t_selected = tf.squeeze(
            diversity_q_t, axis=len(diversity_q_t.shape) - 1)
        if policy.config["twin_q"]:
            twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
        q_tp1 -= model.alpha * log_pis_tp1
        diversity_q_tp1 -= model.alpha * log_pis_tp1

        q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
        q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.DONES],
                                           tf.float32)) * q_tp1_best

        diversity_q_tp1_best = tf.squeeze(
            input=diversity_q_tp1, axis=len(diversity_q_tp1.shape) - 1)
        diversity_q_tp1_best_masked = (1.0 - tf.cast(
            train_batch[SampleBatch.DONES], tf.float32
        )) * q_tp1_best

    assert policy.config["n_step"] == 1, "TODO(hartikainen) n_step > 1"

    # compute RHS of bellman equation
    q_t_selected_target = tf.stop_gradient(
        train_batch[SampleBatch.REWARDS] +
        policy.config["gamma"] ** policy.config["n_step"] * q_tp1_best_masked)

    diversity_q_t_selected_target = tf.stop_gradient(
        train_batch[SampleBatch.REWARDS] +
        policy.config["gamma"] ** policy.config["n_step"] *
        diversity_q_tp1_best_masked)

    # Compute the TD-error (potentially clipped).
    base_td_error = tf.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = tf.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    diversity_td_error = tf.abs(
        diversity_q_t_selected - diversity_q_t_selected_target)

    critic_loss = [
        tf.losses.mean_squared_error(
            labels=q_t_selected_target, predictions=q_t_selected, weights=0.5)
    ]

    diversity_critic_loss = [
        tf.losses.mean_squared_error(
            labels=diversity_q_t_selected_target,
            predictions=diversity_q_t_selected, weights=0.5)
    ]

    if policy.config["twin_q"]:
        critic_loss.append(
            tf.losses.mean_squared_error(
                labels=q_t_selected_target,
                predictions=twin_q_t_selected,
                weights=0.5))

    # Auto-calculate the target entropy.
    if policy.config["target_entropy"] == "auto":
        if model.discrete:
            target_entropy = np.array(-policy.action_space.n, dtype=np.float32)
        else:
            target_entropy = -np.prod(policy.action_space.shape)
    else:
        target_entropy = policy.config["target_entropy"]

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        raise NotImplementedError()
        alpha_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(
                    tf.stop_gradient(policy_t), -model.log_alpha *
                                                tf.stop_gradient(
                                                    log_pis_t +
                                                    target_entropy)),
                axis=-1))
        actor_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t,
                    model.alpha * log_pis_t - tf.stop_gradient(q_t)),
                axis=-1))
    else:
        alpha_loss = -tf.reduce_mean(
            model.log_alpha * tf.stop_gradient(log_pis_t + target_entropy))
        actor_loss = tf.reduce_mean(model.alpha * log_pis_t - q_t_det_policy)
        diversity_actor_loss = tf.reduce_mean(
            model.alpha * log_pis_t - diversity_q_t_det_policy)

    # save for stats function
    policy.q_t = q_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.alpha_value = model.alpha
    policy.target_entropy = target_entropy

    policy.diversity_critic_loss = diversity_critic_loss
    policy.diversity_actor_loss = diversity_actor_loss

    # add what we need here
    policy.diversity_reward_mean = tf.reduce_mean(
        train_batch["diversity_rewards"]
    )

    # in a custom apply op we handle the losses separately, but return them
    # combined in one loss for now
    return actor_loss + tf.add_n(
        critic_loss) + alpha_loss + diversity_critic_loss + diversity_actor_loss


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

    # This part we get the diversity gradient
    if policy.config["grad_norm_clipping"] is not None:
        diversity_actor_grads_and_vars = minimize_and_clip(
            optimizer,
            policy.diversity_actor_loss,
            var_list=policy.model.policy_variables(),
            clip_val=policy.config["grad_norm_clipping"])
        diversity_critic_grads_and_vars = minimize_and_clip(
            optimizer,
            policy.diversity_critic_loss[0],
            var_list=policy.model.diversity_q_variables(),
            clip_val=policy.config["grad_norm_clipping"])
    else:
        diversity_actor_grads_and_vars = \
            policy._actor_optimizer.compute_gradients(
                policy.diversity_actor_loss,
                var_list=policy.model.policy_variables())
        diversity_critic_grads_and_vars = policy._critic_optimizer[
            0].compute_gradients(
            policy.diversity_critic_loss[0],
            var_list=policy.model.diversity_q_variables())

    policy_grad = actor_grads_and_vars
    diversity_grad = diversity_actor_grads_and_vars

    # if policy.config["grad_norm_clipping"] is not None:
    #     diversity_grad = minimize_and_clip(
    #         optimizer, policy.diversity_loss,
    #         var_list=policy.model.policy_variables(),
    #         clip_val=policy.config["grad_norm_clipping"]
    #     )
    # else:
    #     diversity_grad = policy._actor_optimizer.compute_gradients(
    #         policy.diversity_loss, var_list=policy.model.policy_variables()
    #     )

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
        actor_grads_and_vars_fused = [(g, return_gradients[var][1])
                                      for g, (_, var) in
                                      zip(clipped_grads, policy_grad)]
    else:
        actor_grads_and_vars_fused = [
            return_gradients[var] for _, var in policy_grad
        ]

    # save these for later use in build_apply_op
    policy._actor_grads_and_vars = [(g, v) for (g, v) in
                                    actor_grads_and_vars_fused
                                    if g is not None]
    policy._critic_grads_and_vars = [(g, v) for (g, v) in critic_grads_and_vars
                                     if g is not None]
    policy._diversity_critic_grads_and_vars = [(g, v) for (g, v) in
                                               diversity_critic_grads_and_vars
                                               if g is not None]
    policy._alpha_grads_and_vars = [(g, v) for (g, v) in alpha_grads_and_vars
                                    if g is not None]

    grads_and_vars = (
            policy._actor_grads_and_vars +
            policy._critic_grads_and_vars +
            policy._diversity_critic_grads_and_vars +
            policy._alpha_grads_and_vars
    )

    return grads_and_vars
