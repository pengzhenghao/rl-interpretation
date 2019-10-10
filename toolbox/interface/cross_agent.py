"""
This file contain the interface to cross-agent analysis.
"""
import copy
from collections import OrderedDict
from collections.abc import Iterable as IterableClass

import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.cluster import DBSCAN

from toolbox.cluster.process_cluster import ClusterFinder
from toolbox.evaluate.replay import agent_replay
from toolbox.evaluate.symbolic_agent import SymbolicAgentBase
from toolbox.represent.process_fft import stack_fft, parse_df
from toolbox.represent.process_similarity import get_cka

DEFAULT_CONFIG = {"num_samples": 100, "pca_dim": 50}


def get_kl_divergence(dist1, dist2):
    assert dist1.ndim == 2
    assert dist2.ndim == 2

    source_mean, source_log_std = np.split(dist1, 2, axis=1)
    target_mean, target_log_std = np.split(dist2, 2, axis=1)

    kl_divergence = np.sum(
        target_log_std - source_log_std +
        (np.square(source_log_std) + np.square(source_mean - target_mean)) /
        (2.0 * np.square(target_log_std) + 1e-9) - 0.5,
        axis=1
    )
    kl_divergence = np.clip(kl_divergence, 0.0, 1e38)  # to avoid inf
    averaged_kl_divergence = np.mean(kl_divergence)
    return averaged_kl_divergence


def _parse_prediction_to_precision(prediction, agent_info_dict):
    parent_cluster_dict = {}
    correct_predict = 0

    for name, pred_info in prediction.items():
        parent = agent_info_dict[name]['parent']
        parent_cluster_dict[parent] = None

    for parent_name in parent_cluster_dict.keys():
        parent_name_real_name = parent_name + " child=0"
        parent_cluster_dict[parent_name] = \
            prediction[parent_name_real_name]['cluster']

    for name, pred_info in prediction.items():
        parent = agent_info_dict[name]['parent']
        parent_id = parent_cluster_dict[parent]
        predict_id = pred_info['cluster']
        correct_predict += int(predict_id == parent_id)
    precision = correct_predict / len(prediction)
    return precision, parent_cluster_dict, correct_predict


def get_k_means_clustering_precision(representation_dict, agent_info_dict):
    num_agents = len(representation_dict)

    best_precision = float('-inf')
    best_prediction, best_parent_cluster_dict = None, None
    best_correct_predict = 0

    cluster_df = pd.DataFrame(representation_dict).T
    for i in range(5):

        cluster_finder = ClusterFinder(cluster_df, max_num_cluster=5)
        cluster_finder.set(num_agents)

        prediction = cluster_finder.predict()

        precision, parent_cluster_dict, correct_predict = \
            _parse_prediction_to_precision(
                prediction, agent_info_dict)

        if precision > best_precision:
            best_precision = precision
            best_prediction, best_parent_cluster_dict = \
                prediction, parent_cluster_dict
            best_correct_predict = correct_predict

    print(
        "Precision: {}. Identical Cluster {}/{}. Different Parent Cluster {} "
        "[total {}].".format(
            best_precision, best_correct_predict, len(best_prediction),
            set(best_parent_cluster_dict.values()), num_agents
        )
    )
    return best_precision, best_prediction, best_parent_cluster_dict, \
           cluster_df


def get_dbscan_precision(
        agent_info_dict, matrix, eps, min_samples, num_parent,
):
    clustering = DBSCAN(
        eps=eps, min_samples=min_samples, metric="precomputed"
    ).fit_predict(matrix)

    prediction = {
        k: {
            "cluster": c
        }
        for k, c in zip(agent_info_dict.keys(), clustering)
    }

    precision, parent_cluster_dict, correct_predict = \
        _parse_prediction_to_precision(prediction, agent_info_dict)

    cluster_set = set(parent_cluster_dict.values())

    num_agents = len(agent_info_dict)
    trust = False
    if precision > (num_parent / num_agents) and \
            (len(cluster_set) >= num_parent) and \
            (-1 not in cluster_set):
        trust = True

    return precision, prediction, parent_cluster_dict, trust


def grid_search_dbscan_cluster(agent_info_dict, matrix, soft=False, search=30):
    # TODO we can add a binary search here.

    best_precision = float('-inf')
    best_prediction = None
    best_parent_cluster_dict = None

    for min_samples in [1, 2, 3]:

        count = 0
        eps_candidates = np.linspace(1e-6, matrix.max(), 10).tolist()

        while count < search and eps_candidates:
            eps = eps_candidates.pop(0)
            precision, prediction, parent_cluster_dict, trust = \
                get_dbscan_precision(
                agent_info_dict, matrix, eps, min_samples, soft
            )
            count += 1

            if trust and precision > best_precision:
                best_precision = precision
                best_prediction = prediction
                best_parent_cluster_dict = parent_cluster_dict
    if best_prediction is None:
        return 0.0, prediction, parent_cluster_dict
    return best_precision, best_prediction, best_parent_cluster_dict


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
        "similarity": ["cka"],
        "distance": [
            "sunhao", "js", "cka_reciprocal", "cka", "naive", "naive_pca",
            "naive_l1", "fft", "fft_pca"
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

    def feed(self, agent_rollout_dict, name_agent_info_mapping):
        """
        1. store the data
        2. build the joint dataset
        3. lazy's replay the necessary rollout

        :return:
        """

        if self.initialized:
            raise ValueError(
                "The CrossAgentAnalyst should only be initialized once!"
                "But you have fed data before with keys: {}."
                "Please re-instantialized CrossAgentAnalyst.".format(
                    self.agent_rollout_dict.keys()
                )
            )

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
        self.parent_agent_names = [
            name for name in self.agent_rollout_dict.keys()
            if name.endswith("child=0")
        ]
        self.parent_agent_indices = [
            i for i, name in enumerate(self.agent_rollout_dict.keys())
            if name.endswith("child=0")
        ]
        self.name_index_mapping = {
            k: i for i, k in enumerate(self.agent_rollout_dict.keys())
        }

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
        self.name_agent_info_mapping = name_agent_info_mapping.copy()

    def naive_representation(self):
        if self.computed_results['representation']['naive'] is not None:
            return self.computed_results['representation']['naive']

        agent_naive_represent_dict = OrderedDict()
        if self.agent_replay_dict is None:
            raise ValueError(
                "You have not replay all agents on the joint dataset yet! "
                "Please call "
                "CrossAgentAnalyst.replay(name_agent_info_mapping or "
                "name_symbolic_agent_mapping) to replay!"
            )

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
                iterable=list(
                    self.computed_results['representation']['naive'].values()),
                apply_function=lambda x, y: np.linalg.norm(x - y, ord=1)
            )

        self.computed_results['distance']['naive_pca'] = \
            self._get_distance_from_representation(
                self.computed_results['representation']['naive_pca'])

        print(
            "Successfully finish representation.naive / "
            "representation.naive_pca / distance.naive / distance.naive_pca"
        )

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
                df = stack_fft(
                    obs, act, 'std', padding_value=0.0, padding_length=500
                )
                data_frame = df if data_frame is None else data_frame.append(
                    df, ignore_index=True
                )
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
            "/ distance.fft / distance.fft_pca"
        )
        return agent_fft_represent_dict

    def _reduce_dimension(self, input_dict):
        assert input_dict.keys() == self.agent_rollout_dict.keys()

        repr_list_tmp = list(input_dict.values())
        if self.config['pca_dim'] > len(repr_list_tmp):
            print(
                "[WARNING] the pca_dim should not less than "
                "num_samples!!! You call for pca_dim {}, "
                "but only have {} samples.".format(
                    self.config['pca_dim'], len(repr_list_tmp)
                )
            )
        pca_dim = min(self.config['pca_dim'], len(repr_list_tmp))
        pca_result = decomposition.PCA(pca_dim).fit_transform(
            np.stack(repr_list_tmp)
        )
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
        flatten = [
            v['behaviour_logits'] for v in self.agent_replay_info_dict.values()
        ]

        for i1 in range(len(flatten) - 1):
            source = flatten[i1][i1 * num_samples:(i1 + 1) * num_samples]
            for i2 in range(i1, len(flatten)):
                target = flatten[i2][i2 * num_samples:(i2 + 1) * num_samples]
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
                ) + get_kl_divergence(target, average_distribution_target)

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
        self.computed_results['distance']['sunhao'] = sunhao_matrix
        return sunhao_matrix

    # Cluster existing data
    def cluster_representation(self):

        representation_precision_dict = OrderedDict()
        representation_prediction_dict = OrderedDict()
        representation_parent_cluster_dict = OrderedDict()
        cluster_df_dict = OrderedDict()

        agent_info_dict = {
            k: agent.agent_info
            for k, agent in self.name_agent_info_mapping.items()
        }

        for method_name, repr_dict in \
                self.computed_results['representation'].items():
            precision, prediction, parent_cluster_dict, cluster_df = \
                get_k_means_clustering_precision(repr_dict, agent_info_dict)
            representation_precision_dict[method_name] = precision
            representation_prediction_dict[method_name] = copy.deepcopy(
                prediction
            )
            representation_parent_cluster_dict[method_name] = copy.deepcopy(
                parent_cluster_dict
            )
            cluster_df_dict[method_name] = copy.deepcopy(cluster_df)

        self.cluster_representation_cluster_df_dict = cluster_df_dict
        self.cluster_representation_prediction_dict = \
            representation_prediction_dict
        self.cluster_representation_precision_dict = \
            representation_precision_dict
        self.cluster_representation_parent_cluster_dict = \
            representation_parent_cluster_dict

        return representation_precision_dict, \
               representation_prediction_dict, \
               representation_parent_cluster_dict, \
               cluster_df_dict

    def cluster_distance(self):
        agent_info_dict = {
            k: agent.agent_info
            for k, agent in self.name_agent_info_mapping.items()
        }

        precision_dict = OrderedDict()
        prediction_dict = OrderedDict()
        parent_cluster_dict = OrderedDict()

        for method_name, matrix in self.computed_results['distance'].items():
            print('matrix form: ', type(matrix), matrix.shape)
            precision, prediction, parent_cluster = \
                grid_search_dbscan_cluster(agent_info_dict, matrix)
            precision_dict[method_name] = precision
            prediction_dict[method_name] = prediction
            parent_cluster_dict[method_name] = parent_cluster

        self.cluster_distance_precision_dict = precision_dict
        self.cluster_distance_prediction_dict = prediction_dict
        self.cluster_distance_parent_cluster_dict = parent_cluster_dict

        return precision_dict, prediction_dict, parent_cluster_dict

    # Some Public APIs
    def fft(self):
        self.fft_representation()

    def naive(self):
        self.naive_representation()

    def sunhao(self):
        self.sunhao_distance()

    def js(self):
        self.js_distance()

    def cka(self):
        self.cka_similarity()

    def get(self):
        return self.computed_results

    def walkthrough(self):
        self.fft()
        self.naive()
        self.js()
        self.sunhao()
        self.cka()
        return self.get()

    # def cluster_representation(self):
    #     cluster_representation()

    def agent_level_summary(self):
        """
        {
            method-class-name: {
                method1: one_value
                method2: one_value
            }
        }
        """
        dataframe = []

        for name, roll_list in self.agent_rollout_dict.items():
            # mean length
            episode_length = [len(rollout['trajectory']) for rollout in
                              roll_list]

            dataframe.append({
                "label": "episode_length_mean",
                "value": np.mean(episode_length),
                "agent": name
            })

            dataframe.append({
                "label": "episode_length_std",
                "value": np.std(episode_length),
                "agent": name
            })

            episode_reward = [
                sum([transition[-2] for transition in rollout['trajectory']])
                for rollout in roll_list
            ]

            dataframe.append({
                "label": "episode_reward_mean",
                "value": np.mean(episode_reward),
                "agent": name
            })

            dataframe.append({
                "label": "episode_reward_std",
                "value": np.std(episode_reward),
                "agent": name
            })

        for my_id, (name, agent) in enumerate(
                self.name_agent_info_mapping.items()):
            parent_real_name = agent.agent_info['parent'] + " child=0"
            parent_index = self.name_index_mapping[parent_real_name]

            for method_name, matrix in self.computed_results[
                'distance'].items():
                distance = matrix[my_id, parent_index]
                dataframe.append({
                    "label": "{}_to_parent".format(method_name),
                    "value": distance,
                    "agent": name
                })

                dataframe.append({
                    "label": "{}_mean".format(method_name),
                    "value": np.mean(matrix[my_id]),
                    "agent": name
                })

                dataframe.append({
                    "label": "{}_std".format(method_name),
                    "value": np.std(matrix[my_id]),
                    "agent": name
                })
        return pd.DataFrame(dataframe)

    def summary(self):
        agent_level_summary = self.agent_level_summary()
        return_dict = dict()
        return_dict['agent_level_summary'] = agent_level_summary
        return_dict['computed_result'] = copy.deepcopy(self.get())
        return_dict['representation'] = \
            return_dict['computed_result']['representation']

        metric_value_dict = \
            agent_level_summary.groupby('label').mean().to_dict()['value']
        # metric_value_dict['']
        return_dict['metric'] = metric_value_dict

        return_dict['cluster_representation'] = {
            "cluster_df_dict": self.cluster_representation_cluster_df_dict,
            "prediction_dict": self.cluster_representation_prediction_dict
        }

        return_dict['cluster_result'] = {

        }

        return return_dict
