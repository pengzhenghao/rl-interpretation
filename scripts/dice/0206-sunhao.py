import argparse
import os
import pickle
from collections import deque

from ray import tune
from ray.rllib.agents.es.es import *
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import FilterManager

from toolbox import initialize_ray, get_local_dir


class ESTrainer2(ESTrainer):
    def _init(self, config, env_creator):
        policy_params = {"action_noise_std": 0.01}

        env = env_creator(config["env_config"])
        from ray.rllib import models
        preprocessor = models.ModelCatalog.get_preprocessor(env)

        self.sess = utils.make_session(single_threaded=False)
        self.policy = policies.GenericPolicy(
            self.sess, env.action_space, env.observation_space, preprocessor,
            config["observation_filter"], config["model"], **policy_params)
        self.optimizer = optimizers.Adam(self.policy, config["stepsize"])
        self.report_length = config["report_length"]

        # Create the shared noise table.
        logger.info("Creating shared noise table.")
        noise_id = create_shared_noise.remote(config["noise_size"])
        self.noise = SharedNoiseTable(ray.get(noise_id))

        # Create the actors.
        logger.info("Creating actors.")
        self._workers = [
            Worker.remote(config, policy_params, env_creator, noise_id)
            for _ in range(config["num_workers"])
        ]

        self.episodes_so_far = 0
        self.reward_list = []
        self.succ_list = deque(maxlen=100)
        self.tstart = time.time()

    def _train(self):
        config = self.config

        theta = self.policy.get_weights()
        assert theta.dtype == np.float32

        # Put the current policy weights in the object store.
        theta_id = ray.put(theta)
        # Use the actors to do rollouts, note that we pass in the ID of the
        # policy weights.
        results, num_episodes, num_timesteps = self._collect_results(
            theta_id, config["episodes_per_batch"], config["train_batch_size"])

        all_noise_indices = []
        all_training_returns = []
        all_training_lengths = []
        all_eval_returns = []
        all_eval_lengths = []

        # Loop over the results.
        for result in results:
            all_eval_returns += result.eval_returns
            all_eval_lengths += result.eval_lengths

            all_noise_indices += result.noise_indices
            all_training_returns += result.noisy_returns
            all_training_lengths += result.noisy_lengths

        assert len(all_eval_returns) == len(all_eval_lengths)
        assert (len(all_noise_indices) == len(all_training_returns) ==
                len(all_training_lengths))

        self.episodes_so_far += num_episodes

        # Assemble the results.
        eval_returns = np.array(all_eval_returns)
        eval_lengths = np.array(all_eval_lengths)
        noise_indices = np.array(all_noise_indices)
        noisy_returns = np.array(all_training_returns)
        noisy_lengths = np.array(all_training_lengths)

        # Process the returns.
        if config["return_proc_mode"] == "centered_rank":
            proc_noisy_returns = utils.compute_centered_ranks(noisy_returns)
        else:
            raise NotImplementedError(config["return_proc_mode"])

        # Compute and take a step.
        g, count = utils.batched_weighted_sum(
            proc_noisy_returns[:, 0] - proc_noisy_returns[:, 1],
            (self.noise.get(index, self.policy.num_params)
             for index in noise_indices),
            batch_size=500)
        g /= noisy_returns.size
        assert (g.shape == (self.policy.num_params,) and g.dtype == np.float32
                and count == len(noise_indices))
        # Compute the new weights theta.
        theta, update_ratio = self.optimizer.update(-g +
                                                    config["l2_coeff"] * theta)
        # Set the new weights in the local copy of the policy.
        self.policy.set_weights(theta)
        # Store the rewards
        if len(all_eval_returns) > 0:
            self.reward_list.append(np.mean(eval_returns))

            for vv in eval_returns:
                self.succ_list.append(float(bool(vv > -50.0)))

        # Now sync the filters
        FilterManager.synchronize({
            DEFAULT_POLICY_ID: self.policy.get_filter()
        }, self._workers)

        info = {
            "weights_norm": np.square(theta).sum(),
            "grad_norm": np.square(g).sum(),
            "success_rate": np.mean(self.succ_list),
            "update_ratio": update_ratio,
            "episodes_this_iter": noisy_lengths.size,
            "episodes_so_far": self.episodes_so_far,
        }

        reward_mean = np.mean(self.reward_list[-self.report_length:])
        result = dict(
            episode_reward_mean=reward_mean,
            episode_len_mean=eval_lengths.mean(),
            timesteps_this_iter=noisy_lengths.sum(),
            success_rate=np.mean(self.succ_list),
            info=info)

        return result


os.environ['OMP_NUM_THREADS'] = '1'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="FetchPush-v1")
    args = parser.parse_args()

    exp_name = "0206-sunhao-final-{}".format(args.env_name)
    num_gpus = 0


    def on_episode_end(info):
        episode = info['episode']
        if info['episode'].total_reward > -50:
            episode.custom_metrics['success_rate'] = 1
        else:
            episode.custom_metrics['success_rate'] = 0


    initialize_ray(
        test_mode=False,
        local_mode=False,
        num_gpus=num_gpus
    )

    walker_config = {
        "seed": tune.grid_search([i * 100 for i in range(5)]),
        "env": args.env_name,
        "model": {"fcnet_hiddens": [64, 64, 64]},
        'train_batch_size': 2500,
        "episodes_per_batch": 50,

        "num_gpus": 0,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
        'num_workers': 10,
        "callbacks": {
            "on_episode_end": on_episode_end
        }
    }

    analysis = tune.run(
        ESTrainer2,
        local_dir=get_local_dir(),
        name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=10,
        checkpoint_score_attr="episode_reward_mean",
        checkpoint_at_end=True,
        stop={"training_iteration": 600},
        config=walker_config,
        max_failures=20,
        reuse_actors=False,
        verbose=1
    )

    path = "{}.pkl".format(
        exp_name,
    )

    with open(path, "wb") as f:
        data = analysis.fetch_trial_dataframes()
        pickle.dump(data, f)
        print("Result is saved at: <{}>".format(path))
