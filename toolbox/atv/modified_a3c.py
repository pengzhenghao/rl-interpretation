from ray import tune
from ray.rllib.agents.a3c.a3c import A3CTFPolicy, A3CTrainer
from ray.rllib.agents.a3c.a3c_tf_policy import postprocess_advantages


def modified_postprocess(policy, sample_batch, other_batches, episode):
    post_batch = postprocess_advantages(
        policy, sample_batch, other_batches, episode)
    array = post_batch["advantages"]
    post_batch["advantages"] = (array - array.mean()) / max(1e-4, array.std())
    return post_batch


ANA3cTFPolicy = A3CTFPolicy.with_updates(
    name="ANA3CTFPolicy",
    postprocess_fn=modified_postprocess
)

ANA3CTrainer = A3CTrainer.with_updates(
    name="ANA3C",
    default_policy=ANA3cTFPolicy
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
        ANA3CTrainer,
        config=config,
        num_samples=args.num_samples
    )
