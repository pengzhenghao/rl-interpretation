from ray import tune
from ray.rllib.agents.impala.impala import ImpalaTrainer, TreeAggregator
from ray.rllib.optimizers.async_samples_optimizer import AsyncSamplesOptimizer


class ANAsyncSamplesOptimizer(AsyncSamplesOptimizer):
    def _step(self):
        sample_timesteps, train_timesteps = 0, 0

        for train_batch in self.aggregator.iter_train_batches():
            sample_timesteps += train_batch.count

            # Normalize the advantage
            array = train_batch["advantages"]
            train_batch["advantages"] = (array - array.mean()) / max(
                1e-4, array.std())

            self.learner.inqueue.put(train_batch)
            if (self.learner.weights_updated
                    and self.aggregator.should_broadcast()):
                self.aggregator.broadcast_new_weights()

        while not self.learner.outqueue.empty():
            count = self.learner.outqueue.get()
            train_timesteps += count

        return sample_timesteps, train_timesteps


def modified_make_policy_optimizer(workers, config):
    if config["num_aggregation_workers"] > 0:
        # Create co-located aggregator actors first for placement pref
        aggregators = TreeAggregator.precreate_aggregators(
            config["num_aggregation_workers"])
    else:
        aggregators = None
    workers.add_workers(config["num_workers"])

    optimizer = ANAsyncSamplesOptimizer(
        workers,
        lr=config["lr"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        train_batch_size=config["train_batch_size"],
        replay_buffer_num_slots=config["replay_buffer_num_slots"],
        replay_proportion=config["replay_proportion"],
        num_data_loader_buffers=config["num_data_loader_buffers"],
        max_sample_requests_in_flight_per_worker=config[
            "max_sample_requests_in_flight_per_worker"],
        broadcast_interval=config["broadcast_interval"],
        num_sgd_iter=config["num_sgd_iter"],
        minibatch_buffer_size=config["minibatch_buffer_size"],
        num_aggregation_workers=config["num_aggregation_workers"],
        learner_queue_size=config["learner_queue_size"],
        learner_queue_timeout=config["learner_queue_timeout"],
        **config["optimizer"])

    if aggregators:
        # Assign the pre-created aggregators to the optimizer
        optimizer.aggregator.init(aggregators)
    return optimizer


ANIMPALA = ImpalaTrainer.with_updates(
    name="ANIMPALA",
    make_policy_optimizer=modified_make_policy_optimizer,
)

if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, default="")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"env": "BipedalWalker-v2", "num_gpus": 0}

    tune.run(
        ANIMPALA,
        config=config,
        num_samples=args.num_samples
    )
