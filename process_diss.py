from process_cluster import ClusterFinder

import ray
import pandas

@ray.remote
class DissectWorker(object):
    def __init__(self):
        pass

    @ray.method(num_return_vals=0)
    def reset(self,
              run_name,
              ckpt,
              env_name,
              env_maker,
              agent_name,
              padding=None,
              padding_length=None,
              padding_value=None,
              worker_name=None,
              ):
        pass

    @ray.method(num_return_vals=2)
    def dissect(self):
        pass
        return None, None


def parse_representation_dict(representation_dict, *args, **kwargs):
    cluster_df = pandas.DataFrame.from_dict(representation_dict).T
    return cluster_df


def get_diss_representation(
        name_ckpt_mapping, run_name, env_name, env_maker, num_seeds,
        num_rollouts, *args, **kwargs
):
    # Input: a batch of agent, Output: a batch of representation
    pass

def get_dissect_cluster_finder():
    cluster_df = None
    cf = ClusterFinder(cluster_df)
    return cf


if __name__ == '__main__':
    # test codes here.