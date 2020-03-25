from ray import tune
from ray.rllib.agents.a3c.a3c import A3CTFPolicy, A3CTrainer
from ray.rllib.agents.a3c.a3c_tf_policy import postprocess_advantages


def modified_postprocess(policy, sample_batch, other_batches, episode):
    post_batch = postprocess_advantages(
        policy, sample_batch, other_batches, episode)
    array = post_batch["advantages"]
    post_batch["advantages"] = (array - array.mean()) / max(1e-4, array.std())

    print("***********ACTION*********")
    print("action batch max {}, min {}, mean {}".format(
        post_batch["actions"].max(), post_batch["actions"].min(),
        post_batch["actions"].mean()
    ))
    print("***********ACTION*********")

    return post_batch


def get_policy_class_modified(config):
    if config["use_pytorch"]:
        raise NotImplementedError()
    else:
        return ANA3CTFPolicy


ANA3CTFPolicy = A3CTFPolicy.with_updates(
    name="ANA3CTFPolicy",
    postprocess_fn=modified_postprocess
)

ANA3CTrainer = A3CTrainer.with_updates(
    name="ANA3C",
    default_policy=ANA3CTFPolicy,
    get_policy_class=get_policy_class_modified
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
        config = {"env": "BipedalWalker-v2"}

    tune.run(
        ANA3CTrainer,
        config=config,
        num_samples=args.num_samples
    )
