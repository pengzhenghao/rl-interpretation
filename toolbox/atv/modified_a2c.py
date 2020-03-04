from ray import tune
from ray.rllib.agents.a3c.a2c import A2CTrainer, SyncSamplesOptimizer


def choose_policy_optimizer_modified(workers, config):
    if config["microbatch_size"]:
        raise ValueError()
    else:
        return SyncSamplesOptimizer(
            workers,
            train_batch_size=config["train_batch_size"],
            standardize_fields=["advantages"]  # <<== Here!
        )


ANA2CTrainer = A2CTrainer.with_updates(
    name="ANA2C",
    make_policy_optimizer=choose_policy_optimizer_modified
)

if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, default="")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    if args:
        with open(args.file, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    tune.run(
        ANA2CTrainer,
        config=config,
        num_samples=args.num_samples
    )
