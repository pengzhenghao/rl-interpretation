import argparse
import copy
import os
import pickle

from ray import tune
from ray.rllib.agents.a3c import A2CTrainer, A3CTrainer
# from ray.rllib.agents.ddpg import TD3Trainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer

from toolbox import initialize_ray
from toolbox.evaluate import restore_agent
from toolbox.evolution.modified_ars import GaussianARSTrainer
from toolbox.evolution.modified_es import GaussianESTrainer

os.environ['OMP_NUM_THREADS'] = '1'


# def set_td3_from_ppo(td3_agent, ppo_agent_weights):
#     ppo_weights = ppo_agent_weights["default_policy"]
#
#     # modify keys
#     ppo_weights = {
#         k.split('default_policy/')[-1]: v
#         for k, v in ppo_weights.items()
#         if "value" not in k
#     }
#
#     # the output is deterministic
#     # if ppo_agent.get_policy().dist_class is DiagGaussian:
#     tmp_ppo_weights = copy.deepcopy(ppo_weights)
#     for k, v in ppo_weights.items():
#         if "out" in k:
#             if v.ndim == 2:
#                 new_v = np.split(v, 2, axis=1)[0]
#             elif v.ndim == 1:
#                 new_v = np.split(v, 2, axis=0)[0]
#             else:
#                 assert False
#             tmp_ppo_weights[k] = new_v
#     ppo_weights = tmp_ppo_weights
#     # else:
#     #     pass
#
#     key_map = {
#         "dense": "fc_1",
#         "dense_1": "fc_2",
#         "dense_2": "fc_out"
#     }
#
#     td3_weights = td3_agent.get_policy().get_weights()
#     for k, v in td3_weights.items():
#         if "/policy/" in k or "/target_policy/" in k:
#             # k: "default_policy/policy/dense/bias"
#
#             k1 = k.split("/")
#             # k1: ['default_policy', 'policy', 'dense', 'bias']
#             assert k1[2] in key_map
#
#             k2 = "/".join([key_map[k1[2]], *k1[3:]])
#             # k2: 'default_policy/fc_1/bias'
#             assert k2 in ppo_weights, (k2, ppo_weights.keys())
#             assert td3_weights[k].shape == ppo_weights[k2].shape, \
#                 (k, k2, td3_weights[k].shape, ppo_weights[k2].shape)
#             td3_weights[k] = ppo_weights[k2]
#
#     td3_agent.get_policy().set_weights(td3_weights)
#     return td3_agent


def get_dynamic_trainer(algo, init_seed, env_name):
    # if algo == "TD3":
    #     base = TD3Trainer
    if algo == "PPO":
        base = PPOTrainer
    elif algo == "ES":
        # base = ESTrainer
        base = GaussianESTrainer
    elif algo == "A2C":
        base = A2CTrainer
    elif algo == "A3C":
        base = A3CTrainer
    elif algo == "IMPALA":
        base = ImpalaTrainer
    elif algo == "ARS":
        base = GaussianARSTrainer
    else:
        raise NotImplementedError()

    # Create the reference agent.
    ppo_agent = restore_agent("PPO", None, env_name, {
        "seed": init_seed,
        "num_workers": 0
    })

    reference_agent_weights = copy.deepcopy(ppo_agent.get_weights())

    name = "Seed{}-{}".format(init_seed, algo)

    class TrainerWrapper(base):
        _name = name

        def __init__(self, config, *args, **kwargs):
            assert "env" in config, config.keys()
            org_config = copy.deepcopy(base._default_config)
            # our_es = algo == "ES" and base._name == "GaussianES"

            # Update the config if necessary.
            org_config.update(config)
            config = copy.deepcopy(org_config)
            # if algo == "TD3":
            #     config.update({
            #         "actor_hiddens": [256, 256],
            #         "critic_hiddens": [256, 256],
            #         "actor_hidden_activation": "tanh",
            #         "critic_hidden_activation": "tanh"
            #     })
            # elif algo in ["A2C", "A3C", "IMPALA"] or our_es:

            if config["model"]["vf_share_layers"]:
                print(
                    "A2C/A3C/IMPALA should not share value function "
                    "layers. "
                    "So we set config['model']['vf_share_layers'] to "
                    "False")
                config["model"]["vf_share_layers"] = False

            # config["seed"] = init_seed
            # Restore the training agent.

            super().__init__(config, *args, **kwargs)

            self._reference_agent_weights = copy.deepcopy(
                reference_agent_weights
            )

            # Set the weights of the training agent.
            if algo in ["PPO", "A2C", "A3C", "IMPALA", "ES", "ARS"]:
                self.set_weights(self._reference_agent_weights)
            # elif algo == "TD3":
            #     set_td3_from_ppo(self, self._reference_agent_weights)
            # elif algo == "ES":  # For modified GaussianES, treat it like PPO.
            #     set_es_from_ppo(self, ppo_agent)
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
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v2")
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
            "num_envs_per_worker": 16,
            "entropy_coeff": 0.001,
            "lambda": 0.95,
            "lr": 2.5e-4,
        },
        # "TD3": {
        #     "actor_lr": 0.0005,
        #     "buffer_size": 100000,
        #     "actor_hiddens": [256, 256],
        #     "critic_hiddens": [256, 256],
        #     "actor_hidden_activation": "tanh",
        #     "critic_hidden_activation": "tanh"
        # },
        "ES": {"model": {"vf_share_layers": False}},
        "ARS": {"model": {"vf_share_layers": False}},
        "A2C": {
            # "grad_clip": 10.0,
            "num_envs_per_worker": 8,
            # "entropy_coeff": 0.0,
            "lr": 1e-5,
            "model": {"vf_share_layers": False}
        },
        "A3C": {
            # "grad_clip": 10.0,
            "num_envs_per_worker": 8,
            # "entropy_coeff": 0.0,
            "lr": 1e-5,
            "model": {"vf_share_layers": False}
        },
        "IMPALA": {
            # "grad_clip": 10.0,
            "num_envs_per_worker": 8,
            # "entropy_coeff": 0.0,
            "lr": 5e-5,
            "model": {"vf_share_layers": False}
        },
    }

    algo_specify_stop = {
        "PPO": 1e7,
        # "TD3": 1e6,
        "ES": 5e8,
        "ARS": 5e8,
        "A2C": 1e8,
        "A3C": 1e8,
        "IMPALA": 1e8
    }

    stop = int(algo_specify_stop[algo]) if not test else 10000
    config = algo_specify_config[algo]
    config.update({
        "log_level": "DEBUG" if test else "ERROR",
        "num_gpus": 1,  # run 5 experiments in 4 card machine
        "num_cpus_for_driver": 1,
        "num_cpus_per_worker": 1,
        "num_workers": 8
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
