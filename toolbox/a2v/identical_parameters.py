import argparse
import copy
import os

from ray import tune
from ray.rllib.agents.ddpg import TD3Trainer
from ray.rllib.agents.es import ESTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models.tf.tf_action_dist import DiagGaussian

from toolbox import initialize_ray
from toolbox.dece.utils import *
from toolbox.evaluate import restore_agent


def set_es_from_ppo(es_agent, ppo_agent):
    weights = ppo_agent.get_policy().get_weights()

    # modify keys
    weights = {
        k.split('default_policy/')[-1]: v
        for k, v in weights.items()
        if "value" not in k
    }

    # the output is deterministic
    if ppo_agent.get_policy().dist_class is DiagGaussian:
        new_weights = {}
        for k, v in weights.items():
            if "out" in k:
                if v.ndim == 2:
                    new_v = np.split(v, 2, axis=1)[0]
                elif v.ndim == 1:
                    new_v = np.split(v, 2, axis=0)[0]
                else:
                    raise ValueError()
                new_weights[k] = new_v
            else:
                new_weights[k] = v
    else:
        new_weights = weights

    # rename some variables
    tmp_new_weights = copy.deepcopy(new_weights)
    for k, v in new_weights.items():
        if "out" not in k:
            # in ES, the layer names are: fc1, fc_out,
            # but in PPO, they're fc_1, fc_out.
            assert "fc_" in k, k
            new_k = k.replace("fc_", "fc")
            tmp_new_weights[new_k] = v
    new_weights = tmp_new_weights

    # verification
    es_weights = es_agent.policy.variables.get_weights()
    es_keys = es_weights.keys()
    for k in es_keys:
        assert k in new_weights, (k, new_weights.keys(), es_keys)
        assert es_weights[k].shape == new_weights[k].shape, \
            (k, es_weights[k].shape, new_weights[k].shape)

    es_agent.policy.variables.set_weights(new_weights)

    return es_agent


def set_td3_from_ppo(td3_agent, ppo_agent):
    ppo_weights = ppo_agent.get_policy().get_weights()

    # modify keys
    ppo_weights = {
        k.split('default_policy/')[-1]: v
        for k, v in ppo_weights.items()
        if "value" not in k
    }

    # the output is deterministic
    if ppo_agent.get_policy().dist_class is DiagGaussian:
        tmp_ppo_weights = copy.deepcopy(ppo_weights)
        for k, v in ppo_weights.items():
            if "out" in k:
                if v.ndim == 2:
                    new_v = np.split(v, 2, axis=1)[0]
                elif v.ndim == 1:
                    new_v = np.split(v, 2, axis=0)[0]
                else:
                    assert False
                tmp_ppo_weights[k] = new_v
        ppo_weights = tmp_ppo_weights
    else:
        pass

    key_map = {
        "dense": "fc_1",
        "dense_1": "fc_2",
        "dense_2": "fc_out"
    }

    td3_weights = td3_agent.get_policy().get_weights()
    for k, v in td3_weights.items():
        if "/policy/" in k or "/target_policy/" in k:
            # k: "default_policy/policy/dense/bias"

            k1 = k.split("/")
            # k1: ['default_policy', 'policy', 'dense', 'bias']
            assert k1[2] in key_map

            k2 = "/".join([key_map[k1[2]], *k1[3:]])
            # k2: 'default_policy/fc_1/bias'
            assert k2 in ppo_weights, (k2, ppo_weights.keys())
            assert td3_weights[k].shape == ppo_weights[k2].shape, \
                (k, k2, td3_weights[k].shape, ppo_weights[k2].shape)
            td3_weights[k] = ppo_weights[k2]

    td3_agent.get_policy().set_weights(td3_weights)
    return td3_agent


class TrainerBaseWrapper:
    def __init__(self, base, config=None, *args, **kwargs):
        assert "init_seed" in config, config.keys()
        assert "env" in config, config.keys()

        init_seed = config.pop("init_seed")
        env_name = config["env"]
        algo = base._name

        self.__name = "Seed{}-{}".format(init_seed, algo)
        org_config = copy.deepcopy(base._default_config)

        # Create the reference agent.
        ppo_agent = restore_agent("PPO", None, env_name, {
            "seed": init_seed,
            "num_workers": 0
        })

        # Update the config if necessary.
        config = copy.deepcopy(config)
        if algo == "TD3":
            config.update({
                "actor_hiddens": [256, 256],
                "critic_hiddens": [256, 256],
                "actor_hidden_activation": "tanh",
                "critic_hidden_activation": "tanh"
            })
        config["seed"] = init_seed
        org_config.update(config)
        config = org_config

        self.__config = config

        # Restore the training agent.
        print("Super: ", super())
        base.__init__(self, config, *args, **kwargs)

        # Set the weights of the training agent.
        if algo == "PPO":
            self.set_weights(ppo_agent.get_weights())
        elif algo == "TD3":
            set_td3_from_ppo(self, ppo_agent)
        elif algo == "ES":
            set_es_from_ppo(self, ppo_agent)
        else:
            raise NotImplementedError("Config is: {}".format(config))

        self._reference_agent = ppo_agent

    @property
    def _name(self):
        return self.__name


def get_dynamic_trainer(algo):
    if algo == "TD3":
        base = TD3Trainer
    elif algo == "PPO":
        base = PPOTrainer
    elif algo == "ES":
        base = ESTrainer
    else:
        raise NotImplementedError()

    class TrainerWrapper(TrainerBaseWrapper, base):
        def __init__(self, config, *args, **kwargs):
            TrainerBaseWrapper.__init__(self, base, config, *args, **kwargs)

    return TrainerWrapper


def train(
        algo,
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

    trainer = get_dynamic_trainer(algo)
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
        max_failures=20,
        reuse_actors=False,
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
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--stop", type=float, default=5e6)
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v2")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    algo = args.algo
    test = args.test

    os.environ['OMP_NUM_THREADS'] = '1'

    algo_specify_config = {
        "PPO": {
            "num_sgd_iter": 10,
            "num_envs_per_worker": 16,
            "entropy_coeff": 0.001,
            "lambda": 0.95,
            "lr": 2.5e-4,
        },
        "TD3": {
            "actor_lr": 0.0005,
            "buffer_size": 100000,
            "actor_hiddens": [256, 256],
            "critic_hiddens": [256, 256],
            "actor_hidden_activation": "tanh",
            "critic_hidden_activation": "tanh"
        },
        "ES": {
            "observation_filter": "NoFilter",
        }
    }

    algo_specify_stop = {
        "PPO": 1e7,
        "TD3": 1e6,
        "ES": 1e9
    }

    stop = int(algo_specify_stop[algo]) if not test else 10000
    config = algo_specify_config[algo]
    config.update({
        "init_seed": 2020,
        "log_level": "DEBUG" if test else "ERROR",
        "num_gpus": 0.2,
        "num_cpus_for_driver": 0.2,
        "num_cpus_per_worker": 1,
    })

    train(
        algo=algo,
        extra_config=config,
        env_name=args.env_name,
        stop=stop,
        exp_name=args.exp_name,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=test
    )
