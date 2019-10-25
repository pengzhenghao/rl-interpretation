import copy
from collections import OrderedDict

from toolbox import initialize_ray
from toolbox.abstract_worker import WorkerBase, WorkerManagerBase
from toolbox.evaluate.symbolic_agent import SymbolicAgentBase

MAX_NUM_ITERS = 100


class _RemoteSymbolicTrainWorker(WorkerBase):
    def __init__(self):
        self.existing_agent = None

    def run(self, symbolic_agent, stop_criterion):
        assert isinstance(stop_criterion, dict)

        if self.existing_agent is None:
            agent = symbolic_agent.get(default_config=True)['agent']
            self.existing_agent = agent
        else:
            agent = symbolic_agent.get(self.existing_agent, default_config=True)['agent']

        result_list = []
        break_flag = False
        for i in range(MAX_NUM_ITERS):
            result = agent.train()
            result_list.append(result)
            for stop_name, stop_val in stop_criterion.items():
                if result[stop_name] > stop_val:
                    print(
                        "After the {}-th iteration, the criterion {} "
                        "has been achieved: current value {:.2f} is greater "
                        "then stop value: {}. So we break the "
                        "training.".format(
                            i + 1, stop_name, result[stop_name], stop_val
                        )
                    )
                    break_flag = True
                    break
            if break_flag:
                break
        agent_weights = agent.get_weights()
        return result_list, copy.deepcopy(agent_weights)


class RemoteSymbolicTrainManager(WorkerManagerBase):
    def __init__(self, num_workers, total_num=None, log_interval=1):
        super(RemoteSymbolicTrainManager, self).__init__(
            num_workers, _RemoteSymbolicTrainWorker, total_num, log_interval,
            "train"
        )

    def train(self, index, symbolic_agent, stop_criterion):
        keys = [
            'episode_reward_max', 'episode_reward_min', 'episode_reward_mean',
            'episode_len_mean', 'episodes_this_iter', 'policy_reward_min',
            'policy_reward_max', 'policy_reward_mean', 'custom_metrics',
            'sampler_perf', 'off_policy_estimator', 'info',
            'timesteps_this_iter', 'done', 'timesteps_total', 'episodes_total',
            'training_iteration', 'experiment_id', 'date', 'timestamp',
            'time_this_iter_s', 'time_total_s', 'pid', 'hostname', 'node_ip',
            'config', 'time_since_restore', 'timesteps_since_restore',
            'iterations_since_restore', 'num_healthy_workers'
        ]
        for key in stop_criterion.keys():
            assert key in keys
        assert isinstance(symbolic_agent, SymbolicAgentBase)
        symbolic_agent.clear()
        self.submit(index, symbolic_agent, stop_criterion)


    def parse_result(self, result):
        """Overwrite the original function"""
        string = "Beginning Reward: {:.3f}, Ending Reward: {:.3f}".format(
            result[0][0]['episode_reward_mean'],
            result[0][-1]['episode_reward_mean']
        )
        return string


if __name__ == '__main__':
    from toolbox.evaluate.symbolic_agent import MaskSymbolicAgent

    num_agents = 20
    num_workers = 2
    num_train_iters = 1

    initialize_ray(num_gpus=0, test_mode=True)

    spawned_agents = OrderedDict()
    for i in range(num_agents):
        name = "a{}".format(i)
        spawned_agents[name] = MaskSymbolicAgent(
            {
                "run_name": "PPO",
                "env_name": "BipedalWalker-v2",
                "name": name,
                "path": None
            }
        )

    rstm = RemoteSymbolicTrainManager(num_workers, len(spawned_agents))
    for name, agent in spawned_agents.items():
        agent.clear()
        rstm.train(name, agent, num_train_iters)

    result = rstm.get_result()
