"""This file provide function that take agent and probe agent as input and
return the FIM embedding of agent."""
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, SampleBatch, \
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, setup_mixins, \
    ValueNetworkMixin

from toolbox.evaluate import restore_agent


class FIMEmbeddingMixin:
    def __init__(self):
        logp = self._action_logp
        logp = tf.reduce_mean(logp)
        variables = [
            v for v in self.model.trainable_variables()
            if "value" not in v.name
        ]
        grad_var_pairs = self.optimizer().compute_gradients(logp, variables)
        embedding = tf.concat(
            [tf.reshape(grad, shape=[-1]) for grad, _ in
             grad_var_pairs], axis=0)
        fim_embedding = tf.square(embedding)

        def get_fim_embedding(ob):
            return self.get_session().run(fim_embedding, feed_dict={
                self._input_dict[SampleBatch.CUR_OBS]: ob
            })

        self.get_fim_embedding = get_fim_embedding


def before_loss_init(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    FIMEmbeddingMixin.__init__(policy)


PPOFIMTFPolicy = PPOTFPolicy.with_updates(
    name="PPOFIMTFPolicy",
    before_loss_init=before_loss_init,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, FIMEmbeddingMixin
    ]
)


def get_policy_class(config):
    if config.get("use_pytorch") is True:
        raise NotImplementedError()
    else:
        return PPOFIMTFPolicy


PPOFIMTrainer = PPOTrainer.with_updates(
    name="PPOFIM",
    default_policy=PPOFIMTFPolicy,
)


def agent_to_vector(target_agent, probe_agent):
    # Step 1: sample a dataset for given subject_agent
    dataset = []
    for i in range(20):
        dataset.append(target_agent.workers.local_worker().sample())
    dataset = dataset[0].concat_samples(dataset)
    dataset.shuffle()
    # TODO not sure the samples is uniformly spread since each batch is
    #  in one episode.

    # Step 2: compute the embdding for subject_agent via probe_agent
    ob = dataset[SampleBatch.CUR_OBS]
    embedding = probe_agent.get_policy().get_fim_embedding(ob)
    return embedding


if __name__ == '__main__':
    from toolbox import initialize_ray

    initialize_ray(test_mode=True, local_mode=True)

    config = {
        "sample_batch_size": 50,
        "num_workers": 0
    }

    target_agent = restore_agent("PPO", None, "BipedalWalker-v2", config)
    probe_agent = restore_agent(PPOFIMTrainer, None, "BipedalWalker-v2",
                                config)
    agent_to_vector(target_agent, probe_agent)
