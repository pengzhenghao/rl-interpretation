"""
This file contain the interface to cross-agent analysis.
"""
from collections import OrderedDict

import numpy as np
from sklearn import decomposition

from toolbox.evaluate.replay import agent_replay
from toolbox.evaluate.symbolic_agent import SymbolicAgentBase
from toolbox.represent.process_fft import stack_fft, parse_df

DEFAULT_CONFIG = {
    "num_samples": 100,
    "pca_dim": 50
}


class CrossAgentAnalyst:
    """
    The logic of this interface:

    1. feed data
    2. try the methods whatever you like

    self.computed_result = {
        "big class of method": {
            "method 1": result,
            "method 2": result
        },
        ...
        "represent": {
            "fft_represent": ...,
            ...
        }
    }

    """

    methods = {
        "representation": ["fft", "naive", "fft_pca", "naive_pca"],
        "similarity": ["cka_similarity"],
        "distance":
            ["js_distance", "cka_distance", "naive_represent_distance"]
    }

    def __init__(self, config=None):

        self.computed_results = {}
        for k, name_list in self.methods.items():
            self.computed_results[k] = {
                method_name: None
                for method_name in name_list
            }

        self.rollout_dataset = None
        self.config = config if config is not None else DEFAULT_CONFIG
        self.initialized = False

    def _check_input(self):
        if self.rollout_dataset is None:
            print(
                "Data is not loaded! Please call feed(...) before "
                "doing anything!"
            )
            return False
        return True

    def feed(self, agent_rollout_dict):
        """
        1. store the data
        2. build the joint dataset
        3. lazy's replay the necessary rollout

        :return:
        """

        if self.initialized:
            print("The CrossAgentAnalyst should only be initialized once!"
                  "But you have fed data before with keys: {}."
                  "Please re-instantialized CrossAgentAnalyst.".format(
                self.agent_rollout_dict.keys()
            ))
            raise ValueError(
                "The CrossAgentAnalyst should only be initialized once!"
                "But you have fed data before with keys: {}."
                "Please re-instantialized CrossAgentAnalyst.".format(
                    self.agent_rollout_dict.keys()
                ))

        self.initialized = True
        # check the agent_rollout_dict's format
        assert isinstance(agent_rollout_dict, dict)
        rollout_list = next(iter(agent_rollout_dict.values()))
        assert isinstance(rollout_list, list)
        assert isinstance(rollout_list[0], dict)
        assert "trajectory" in rollout_list[0]
        assert "extra_info" in rollout_list[0]
        traj = rollout_list[0]['trajectory']
        assert isinstance(traj, list)  # each entry is a timestep
        assert isinstance(traj[0], list)  # each entry is in obs/act/rew/done
        assert isinstance(traj[0][-1], bool)  # this is the done flag.

        # build the joint dataset
        agent_obs_dict = OrderedDict()
        agent_act_dict = OrderedDict()
        num_samples = self.config['num_samples']
        for name, rollout_list in agent_rollout_dict.items():
            obs_list = [
                tran[0] for r in rollout_list for tran in r['trajectory']
            ]
            act_list = [
                tran[1] for r in rollout_list for tran in r['trajectory']
            ]
            assert len(obs_list[0]) == 17
            agent_obs_dict[name] = np.stack(obs_list)
            agent_act_dict[name] = np.stack(act_list)

        joint_act_dataset = []
        joint_obs_dataset = []
        for (name, act), (name2, obs) in zip(agent_act_dict.items(),
                                             agent_obs_dict.items()):
            assert name == name2
            indices = np.random.randint(0, len(act), num_samples)
            joint_act_dataset.append(act[indices].copy())
            joint_obs_dataset.append(obs[indices].copy())
        joint_act_dataset = np.concatenate(joint_act_dataset)
        joint_obs_dataset = np.concatenate(joint_obs_dataset)

        self.agent_rollout_dict = agent_rollout_dict
        self.agent_act_dict = agent_act_dict
        self.agent_obs_dict = agent_obs_dict
        self.joint_act_dataset = joint_act_dataset
        self.joint_obs_dataset = joint_obs_dataset

    def replay(self, name_agent_info_mapping):
        assert name_agent_info_mapping.keys() == self.agent_rollout_dict.keys()
        # replay
        agent_replay_dict = OrderedDict()
        agent_replay_info_dict = OrderedDict()
        for i, (name, agent_info) in \
                enumerate(name_agent_info_mapping.items()):

            if isinstance(agent_info, SymbolicAgentBase):
                agent_info = agent_info.get()

            act = agent_replay(agent_info['agent'], self.joint_obs_dataset)
            agent_replay_dict[name] = act[0]
            agent_replay_info_dict[name] = act[1]

        self.agent_replay_info_dict = agent_replay_info_dict
        self.agent_replay_dict = agent_replay_dict

    def naive_representation(self):
        pass

    def fft_representation(self):
        if self.computed_results['representation']['fft'] is not None:
            return self.computed_results['representation']['fft']

        agent_fft_represent_dict = OrderedDict()
        for name, rollout_list in self.agent_rollout_dict.items():
            data_frame = None
            for roll in rollout_list:
                traj = roll['trajectory']
                obs = np.stack([t[0] for t in traj])
                act = np.stack([t[1] for t in traj])
                df = stack_fft(obs, act, 'std', padding_value=0.0,
                               padding_length=500)
                data_frame = df if data_frame is None else data_frame.append(
                    df, ignore_index=True)
            reprensentation = parse_df(data_frame)
            agent_fft_represent_dict[name] = reprensentation
        self.computed_results['representation']['fft'] = \
            agent_fft_represent_dict

        return agent_fft_represent_dict

    def fft_pca_representation(self):
        if self.computed_results['representation']['fft_pca'] is not None:
            return self.computed_results['representation']['fft_pca']

        if self.computed_results['representation']['fft'] is None:
            self.fft_representation()

        repr_list_tmp = \
            list(self.computed_results['representation']['fft'].values())

        if self.config['pca_dim'] > len(repr_list_tmp):
            print(
                "!!![ERROR] the pca_dim should not less than "
                "num_samples!!!!!!")
        pca_dim = min(self.config['pca_dim'], len(repr_list_tmp))
        fft_pca_result = decomposition.PCA(pca_dim).fit_transform(
            np.stack(repr_list_tmp))

        agent_fft_pca_represent_dict = OrderedDict()
        for i, name in enumerate(self.agent_rollout_dict.keys()):
            agent_fft_pca_represent_dict[name] = fft_pca_result[i]

        self.computed_results['representation'][
            'fft_pca'] = agent_fft_pca_represent_dict
        return agent_fft_pca_represent_dict

    def cka_similarity(self):
        pass
