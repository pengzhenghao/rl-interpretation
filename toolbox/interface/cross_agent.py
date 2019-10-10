"""
This file contain the interface to cross-agent analysis.
"""
from collections import OrderedDict
from collections.abc import Iterable as IterableClass

import numpy as np
from sklearn import decomposition

from toolbox.evaluate.replay import agent_replay
from toolbox.evaluate.symbolic_agent import SymbolicAgentBase
from toolbox.represent.process_fft import stack_fft, parse_df
from toolbox.represent.process_similarity import get_cka
DEFAULT_CONFIG = {
    "num_samples": 100,
    "pca_dim": 50
}


def get_kl_divergence(dist1, dist2):
    assert dist1.ndim == 2
    assert dist2.ndim == 2

    source_mean, source_log_std = np.split(dist1, 2, axis=1)
    target_mean, target_log_std = np.split(dist2, 2, axis=1)

    kl_divergence = np.sum(
        target_log_std - source_log_std +
        (np.square(source_log_std) + np.square(source_mean - target_mean))
        / (2.0 * np.square(target_log_std) + 1e-9) - 0.5,
        axis=1
    )
    kl_divergence = np.clip(kl_divergence, 0.0, 1e38)  # to avoid inf
    averaged_kl_divergence = np.mean(kl_divergence)
    return averaged_kl_divergence


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
        "distance": [
            "sunhao",
            "js",
            "cka_reciprocal",
            "cka",
            "naive",
            "naive_pca",
            "naive_l1",
            "fft",
            "fft_pca"
        ]
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
        if self.computed_results['representation']['naive'] is not None:
            return self.computed_results['representation']['naive']

        agent_naive_represent_dict = OrderedDict()
        if self.agent_replay_dict is None:
            raise ValueError(
                "You have not replay all agents on the joint dataset yet! "
                "Please call "
                "CrossAgentAnalyst.replay(name_agent_info_mapping or "
                "name_symbolic_agent_mapping) to replay!")

        for name, replay_act in self.agent_replay_dict.items():
            agent_naive_represent_dict[name] = replay_act.flatten()

        self.computed_results['representation']['naive'] = \
            agent_naive_represent_dict

        self.computed_results['representation'][
            'naive_pca'] = self._reduce_dimension(agent_naive_represent_dict)

        self.computed_results['distance']['naive'] = \
            self._get_distance_from_representation(
                self.computed_results['representation']['naive'])

        self.computed_results['distance']['naive_l1'] = \
            self._build_matrix(
                iterable=list(self.computed_results['representation']['naive'].values()),
                apply_function = lambda x, y: np.linalg.norm(x - y, ord=1)
            )


        self.computed_results['distance']['naive_pca'] = \
            self._get_distance_from_representation(
                self.computed_results['representation']['naive_pca'])

        print(
            "Successfully finish representation.naive / "
            "representation.naive_pca / distance.naive / distance.naive_pca")

        return agent_naive_represent_dict

    def naive_pca_representation(self):
        if self.computed_results['representation']['naive_pca'] is None:
            self.naive_representation()
        return self.computed_results['representation']['naive_pca']

    def naive_representation_distance(self):
        if self.computed_results['distance']['naive'] is None:
            self.naive_representation()
        return self.computed_results['distance']['naive']

    def naive_pca_distance(self):
        if self.computed_results['distance']['naive_pca'] is None:
            self.naive_representation()
        return self.computed_results['distance']['naive_pca']

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

        self.computed_results['representation'][
            'fft_pca'] = self._reduce_dimension(agent_fft_represent_dict)

        self.computed_results['distance']['fft'] = \
            self._get_distance_from_representation(
                self.computed_results['representation']['fft'])

        self.computed_results['distance']['fft_pca'] = \
            self._get_distance_from_representation(
                self.computed_results['representation']['fft_pca'])

        print(
            "Successfully finish representation.fft / representation.fft_pca "
            "/ distance.fft / distance.fft_pca")
        return agent_fft_represent_dict

    def fft_pca_representation(self):
        if self.computed_results['representation']['fft_pca'] is None:
            self.fft_representation()
        return self.computed_results['representation']['fft_pca']

    def fft_representation_distance(self):
        if self.computed_results['distance']['fft'] is None:
            self.fft_representation()
        return self.computed_results['distance']['fft']

    def fft_pca_distance(self):
        if self.computed_results['distance']['fft_pca'] is None:
            self.fft_representation()
        return self.computed_results['distance']['fft_pca']

    def _reduce_dimension(self, input_dict):
        assert input_dict.keys() == self.agent_rollout_dict.keys()

        repr_list_tmp = list(input_dict.values())
        if self.config['pca_dim'] > len(repr_list_tmp):
            print(
                "[WARNING] the pca_dim should not less than "
                "num_samples!!! You call for pca_dim {}, "
                "but only have {} samples.".format(
                    self.config['pca_dim'], len(repr_list_tmp)))
        pca_dim = min(self.config['pca_dim'], len(repr_list_tmp))
        pca_result = decomposition.PCA(pca_dim).fit_transform(
            np.stack(repr_list_tmp))
        ret = OrderedDict()
        for i, name in enumerate(input_dict.keys()):
            ret[name] = pca_result[i]

        return ret

    def _get_distance_from_representation(self, input_dict):
        assert isinstance(input_dict, OrderedDict)
        assert input_dict.keys() == self.agent_rollout_dict.keys()
        iterable = list(input_dict.values())
        apply_function = lambda x, y: np.linalg.norm(x - y)
        return self._build_matrix(iterable, apply_function)

    def _build_matrix(self, iterable, apply_function, default_value=0):
        """
        This is a convient function to build a sysmetry function.
        """
        assert isinstance(iterable, IterableClass)
        length = len(iterable)
        assert length == len(self.agent_rollout_dict)
        matrix = np.empty((length, length))
        matrix.fill(default_value)
        for i1 in range(length - 1):
            for i2 in range(i1, length):
                repr1 = iterable[i1]
                repr2 = iterable[i2]
                result = apply_function(repr1, repr2)
                matrix[i1, i2] = result
                matrix[i2, i1] = result
        return matrix

    def cka_similarity(self):
        agent_activation_dict = OrderedDict()

        for name, replay_result in self.agent_replay_info_dict.items():
            activation = replay_result['layer1']
            agent_activation_dict[name] = activation

        iterable = list(agent_activation_dict.values())
        apply_function = get_cka
        cka_similarity = self._build_matrix(iterable, apply_function, 1.0)

        self.computed_results['similarity']['cka'] = cka_similarity
        self.computed_results['distance']['cka'] = \
            np.clip(1 - cka_similarity, 0.0, None)
        self.computed_results['distance']['cka_reciprocal'] = \
            1 / (cka_similarity + 1e-9)
        return cka_similarity



    def js_distance(self):
        # https://stats.stackexchange.com/questions/7630/clustering-should-i
        # -use-the-jensen-shannon-divergence-or-its-square
        # square root is a real distance measurement.

        assert isinstance(self.agent_replay_info_dict, OrderedDict)

        num_samples = self.config['num_samples']
        length = len(self.agent_replay_info_dict)
        js_matrix = np.zeros((length, length))
        flatten = [v['behaviour_logits'] for v in
                   self.agent_replay_info_dict.values()]

        for i1 in range(len(flatten) - 1):
            source = flatten[i1][i1 * num_samples: (i1 + 1) * num_samples]
            for i2 in range(i1, len(flatten)):
                target = flatten[i2][i2 * num_samples: (i2 + 1) * num_samples]
                average_distribution_source = \
                    (source +
                     flatten[i2][i1 * num_samples: (i1 + 1) * num_samples]
                     ) / 2
                average_distribution_target = \
                    (target +
                     flatten[i1][i2 * num_samples: (i2 + 1) * num_samples]
                     ) / 2

                js_divergence = get_kl_divergence(
                    source, average_distribution_source
                ) + get_kl_divergence(
                    target, average_distribution_target
                )

                js_divergence = js_divergence / 2
                js_matrix[i1, i2] = js_divergence
                js_matrix[i2, i1] = js_divergence
        js_matrix = np.sqrt(js_matrix)
        self.computed_results['distance']['js'] = js_matrix
        return js_matrix

    def sunhao_distance(self):
        flatten = list(self.agent_replay_dict.values())
        apply_function = lambda x, y: np.linalg.norm(x - y)
        sunhao_matrix = self._build_matrix(flatten, apply_function)
        self.computed_results['distance']['sunhao_distance'] = sunhao_matrix
        return sunhao_matrix

    # Some Public APIs
    def fft(self):
        self.fft_representation()
        return True

    def naive(self):
        self.naive_representation()
        return True

    def sunhao(self):
        self.sunhao_distance()
        return True

    def js(self):
        self.js_distance()
        return True

    def cka(self):
        self.cka_similarity()
        return True

    def get(self):
        return self.computed_results
