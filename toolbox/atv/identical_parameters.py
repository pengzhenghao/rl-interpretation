import argparse
import copy
import os
import pickle

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from toolbox import initialize_ray
from toolbox.atv import ANA2CTrainer, ANA3CTrainer, ANIMPALATrainer
from toolbox.atv.wrapped_env import register
from toolbox.evaluate import restore_agent
from toolbox.evolution.modified_ars import GaussianARSTrainer
from toolbox.evolution.modified_es import GaussianESTrainer

register()

os.environ['OMP_NUM_THREADS'] = '1'


def get_dynamic_trainer(algo, init_seed, env_name):
    if algo == "PPO":
        base = PPOTrainer
    elif algo == "ES":
        base = GaussianESTrainer
    elif algo == "A2C":
        base = ANA2CTrainer
    elif algo == "A3C":
        base = ANA3CTrainer
    elif algo == "IMPALA":
        base = ANIMPALATrainer
    elif algo == "ARS":
        base = GaussianARSTrainer
    else:
        raise NotImplementedError()

    # Create the reference agent.
    ppo_agent = restore_agent("PPO", None, env_name, {
        "seed": init_seed,
        "num_workers": 0
    })

    reference_weights = copy.deepcopy(ppo_agent.get_weights())
    reference_weights_id = ray.put(reference_weights)
    name = "Seed{}-{}".format(init_seed, algo)

    class TrainerWrapper(base):
        _name = name

        def __init__(self, config, *args, **kwargs):
            assert "env" in config, config.keys()
            org_config = copy.deepcopy(base._default_config)

            # Update the config if necessary.
            org_config.update(config)
            config = copy.deepcopy(org_config)

            if config["model"]["vf_share_layers"]:
                print(
                    "A2C/A3C/IMPALA should not share value function "
                    "layers. "
                    "So we set config['model']['vf_share_layers'] to "
                    "False")
                config["model"]["vf_share_layers"] = False

            super().__init__(config, *args, **kwargs)

            self._reference_agent_weights = ray.get(reference_weights_id)

            print("We have received reference agent weights: ",
                  self._reference_agent_weights)

            # Set the weights of the training agent.
            if algo in ["PPO", "A2C", "A3C", "IMPALA", "ES", "ARS"]:
                self.set_weights(self._reference_agent_weights)
            else:
                raise NotImplementedError("Algo is: {}. Config is: {}"
                                          "".format(algo, config))

    TrainerWrapper.__name__ = name
    TrainerWrapper.__qualname__ = name
    return TrainerWrapper


def train(
        algo,
        init_seed,
        extra_config,
        env_name,
        stop,
        exp_name,
        num_seeds,
        num_gpus,
        test_mode=False,
        **kwargs
):
    initialize_ray(test_mode=test_mode, local_mode=False, num_gpus=num_gpus)
    config = {
        "seed": tune.grid_search([i * 100 for i in range(num_seeds)]),
        "env": env_name,
        "log_level": "DEBUG" if test_mode else "INFO"
    }
    if extra_config:
        config.update(extra_config)

    trainer = get_dynamic_trainer(algo, init_seed, env_name)
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=5,
        checkpoint_score_attr="episode_reward_mean",
        checkpoint_at_end=True,
        stop={"timesteps_total": stop}
        if isinstance(stop, int) else stop,
        config=config,
        max_failures=5,
        **kwargs
    )

    path = "{}-{}-{}ts-{}.pkl".format(exp_name, env_name, stop, algo)
    with open(path, "wb") as f:
        data = analysis.fetch_trial_dataframes()
        pickle.dump(data, f)
        print("Result is saved at: <{}>".format(path))

    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--init-seed", type=int, default=2020)
    parser.add_argument("--env-name", type=str,
                        default="WrappedBipedalWalker-v2")
    # parser.add_argument("--env-name", type=str, default="BipedalWalker-v2")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    algo = args.algo
    test = args.test
    exp_name = "{}-{}-initseed{}-{}seeds".format(args.exp_name, algo,
                                                 args.init_seed,
                                                 args.num_seeds)

    algo_specify_config = {
        "PPO": {
            "num_sgd_iter": 10,
            "num_envs_per_worker": 8,
            "entropy_coeff": 0.001,
            "lambda": 0.95,
            "lr": 2.5e-4,
        },
        "ES": {"model": {"vf_share_layers": False}},
        "ARS": {"model": {"vf_share_layers": False}},
        "A2C": {
            # "num_envs_per_worker": 8,
            "model": {"vf_share_layers": False},
            "entropy_coeff": 0.0,
            # "lr": 1e-5
        },
        "A3C": {
            # "num_envs_per_worker": 8,
            # "sample_batch_size": 20,
            # "entropy_coeff": 0.001,
            "entropy_coeff": 0.0,
            # "lr": 1e-5,
            "model": {"vf_share_layers": False}
        },
        "IMPALA": {
            "num_envs_per_worker": 8,
            "entropy_coeff": 0.001,
            "lr": 1e-4,
            "model": {"vf_share_layers": False}
        },
    }

    algo_specify_stop = {
        "PPO": 1e7,
        "ES": 5e8,
        "ARS": 5e8,
        "A2C": 5e7,
        "A3C": 5e7,
        "IMPALA": 5e7
    }

    stop = int(algo_specify_stop[algo])
    config = algo_specify_config[algo]
    config.update({
        "log_level": "DEBUG" if test else "ERROR",
        "num_gpus": 1,  # run 5 experiments in 4 card machine
        "num_cpus_for_driver": 1,
        "num_cpus_per_worker": 1,
        # "num_workers": 8
    })

    if algo in ["ES", "ARS"]:
        config["num_gpus"] = 0
        config["num_cpus_per_worker"] = 0.5
        config["num_workers"] = 10

    train(
        algo=algo,
        init_seed=args.init_seed,
        extra_config=config,
        env_name=args.env_name,
        stop=stop,
        exp_name=exp_name,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=test,
        verbose=1,
    )
