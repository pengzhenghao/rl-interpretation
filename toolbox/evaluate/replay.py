import copy
import logging
import time
from math import ceil

import numpy as np
import ray

from toolbox.abstract_worker import WorkerManagerBase
from toolbox.evaluate.evaluate_utils import restore_agent_with_activation, \
    restore_agent
from toolbox.evaluate.symbolic_agent import SymbolicAgentBase
from toolbox.process_data.process_data import read_yaml
from toolbox.utils import has_gpu

logger = logging.getLogger(__name__)


def _replay(obs, run_name, ckpt, env_name, require_activation=True):
    if require_activation:
        agent = restore_agent_with_activation(run_name, ckpt, env_name)
    else:
        agent = restore_agent(run_name, ckpt, env_name)
    act, infos = agent_replay(agent, obs)
    return act, infos


# @ray.remote
def remote_symbolic_replay(symbolic_agent, obs):
    assert isinstance(symbolic_agent, SymbolicAgentBase)
    agent = symbolic_agent.get()['agent']
    return agent_replay(agent, obs)


class _RemoteSymbolicReplayWorker:
    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None, resources=None):
        return ray.remote(
            num_cpus=num_cpus, num_gpus=num_gpus, resources=resources
        )(cls)

    def __init__(self):
        self.existing_agent = None

    def replay(self, symbolic_agent, obs):
        if self.existing_agent is None:
            agent = symbolic_agent.get()['agent']
            self.existing_agent = agent
        else:
            agent = symbolic_agent.get(self.existing_agent)['agent']
        logger.debug(
            "[_RemoteSymbolicReplayWorker] Start to replay agent <{}>".format(
                symbolic_agent.name
            )
        )
        ret = agent_replay(agent, obs)
        logger.debug(
            "[_RemoteSymbolicReplayWorker] Finish to replay agent <{}>".format(
                symbolic_agent.name
            )
        )
        return ret


class RemoteSymbolicReplayManager(WorkerManagerBase):
    def __init__(self, num_workers, total_num=None, log_interval=1):
        super(RemoteSymbolicReplayManager, self).__init__(
            num_workers, _RemoteSymbolicReplayWorker, total_num, log_interval,
            "replay"
        )

    def replay(self, index, symbolic_agent, obs):
        assert isinstance(symbolic_agent, SymbolicAgentBase)
        symbolic_agent.clear()
        oid = self.workers[self.pointer].replay.remote(symbolic_agent, obs)
        self.postprocess(index, oid)


def agent_replay(agent, obs):
    act, _, infos = agent.get_policy().compute_actions(obs)
    return act, infos


local_replay = _replay


@ray.remote(num_gpus=0.2)
def remote_replay_gpu(obs, run_name, ckpt, env_name):
    return _replay(obs, run_name, ckpt, env_name)


@ray.remote
def remote_replay_cpu(obs, run_name, ckpt, env_name):
    return _replay(obs, run_name, ckpt, env_name)


def several_agent_replay(
        yaml_path,
        obs,
        # num_rollouts,
        seed=0,
        num_workers=10,
        _num_agents=None
        # force_rewrite=False,
        # return_data=False
):
    name_ckpt_mapping = read_yaml(yaml_path, number=_num_agents)
    now_t_get = now_t = start_t = time.time()
    num_agents = len(name_ckpt_mapping)
    num_iteration = int(ceil(num_agents / num_workers))
    agent_ckpt_dict_range = list(name_ckpt_mapping.items())
    agent_count = 1
    agent_count_get = 1

    have_gpu = has_gpu()
    return_dict = {}

    for iteration in range(num_iteration):
        start = iteration * num_workers
        end = min((iteration + 1) * num_workers, num_agents)
        # obj_ids = []
        # workers = []
        obj_ids_dict = {}
        for i, (name, ckpt_dict) in \
                enumerate(agent_ckpt_dict_range[start:end]):
            ckpt = ckpt_dict["path"]
            env_name = ckpt_dict["env_name"]
            # if "env_name" in ckpt_dict else "BipedalWalker-v2"
            # env_maker = ENV_MAKER_LOOKUP[env_name]
            run_name = ckpt_dict["run_name"]
            # if "run_name" in ckpt_dict else "PPO"
            assert run_name == "PPO"

            if have_gpu:
                obj_id = remote_replay_gpu.remote(
                    obs, run_name, ckpt, env_name
                )
            else:
                obj_id = remote_replay_cpu.remote(
                    obs, run_name, ckpt, env_name
                )
            obj_ids_dict[name] = obj_id

            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Start collect replay result"
                " of {} samples from agent <{}>".format(
                    agent_count, num_agents,
                    time.time() - now_t,
                    time.time() - start_t, obs.shape, name
                )
            )

            agent_count += 1
            now_t = time.time()

        for agent_name, obj_id in obj_ids_dict.items():
            act, infos = copy.deepcopy(ray.get(obj_id))
            return_dict[agent_name] = {"act": act, "infos": infos}

            # trajectory_list = []
            # for obj_id in obj_ids:
            #     trajectory_list.append(ray.get(obj_id))
            # return_dict[name] = trajectory_list
            # worker.close.remote()
            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Collected output of {} samples "
                "from agent <{}>".format(
                    agent_count_get, num_agents,
                    time.time() - now_t_get,
                    time.time() - start_t, obs.shape, agent_name
                )
            )
            agent_count_get += 1
            now_t_get = time.time()
    return return_dict


def deprecated_replay(trajectory, agent):
    obs_batch = [tansition[0] for tansition in trajectory]
    obs_batch = np.asarray(obs_batch)
    actions, infos = agent_replay(agent, obs_batch)
    return actions, infos
