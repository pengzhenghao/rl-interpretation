# from process_cluster import ClusterFinder
from collections import OrderedDict

from record_video import GridVideoRecorder


def _build_name_row_mapping(cluster_dict):
    """
    cluster_dict = {name: {"distance": float, "cluster": int}, ..}
    :param cluster_dict:
    :return:
    """
    ret = {}
    for name, cluster_info in cluster_dict.items():
        ret[name] = "Cluster {}".format(cluster_info['cluster'])
    return ret


def _build_name_col_mapping(cluster_dict):
    # We haven't found the suitable names for each column.
    return None


def transform_name_ckpt_mapping(
        name_ckpt_mapping, cluster_dict, max_num_cols=10
):
    # Function:
    # 1. re-order the OrderedDict
    # 2. add distance information in name
    clusters = set([c_info['cluster'] for c_info in cluster_dict.values()])
    num_clusters = len(clusters)

    new_name_ckpt_mapping = OrderedDict()

    old_row_mapping = _build_name_row_mapping(cluster_dict)
    old_col_mapping = _build_name_col_mapping(cluster_dict)

    name_loc_mapping = {}
    name_row_mapping = {} if old_row_mapping is not None else None
    name_col_mapping = {} if old_col_mapping is not None else None

    for row_id, cls in enumerate(clusters):
        within_one_cluster = {
            k: v
            for k, v in cluster_dict.items() if v['cluster'] == cls
        }
        pairs = sorted(
            within_one_cluster.items(), key=lambda kv: kv[1]['distance']
        )
        if len(pairs) > max_num_cols:
            pairs = pairs[:max_num_cols]
        # {'cc': {'distance': 1}, 'aa': {'distance': 10}, ..}
        row_cluster_dict = dict(pairs)
        for col_id, (name, cls_info) in enumerate(row_cluster_dict.items()):
            loc = (row_id, col_id)

            components = name.split(" ")
            new_name = components[0]
            for com in components:
                if "=" not in com:
                    continue
                new_name += "," + com.split('=')[0][0] + com.split('=')[1]

            new_name = new_name + " d{:.2f}".format(cls_info['distance'])
            new_name_ckpt_mapping[new_name] = name_ckpt_mapping[name]
            name_loc_mapping[new_name] = loc
            if old_row_mapping is not None:
                name_row_mapping[new_name] = old_row_mapping[name]
            if old_col_mapping is not None:
                name_col_mapping[new_name] = old_col_mapping[name]

    return new_name_ckpt_mapping, name_loc_mapping, name_row_mapping, \
           name_col_mapping


def generate_video_of_cluster(
        cluster_dict,
        name_ckpt_mapping,
        video_path,
        env_name,
        run_name,
        max_num_cols=10,
        seed=0,
        local_mode=False,
        steps=int(1e10),
        num_workers=5
):
    assert isinstance(cluster_dict, dict)
    assert isinstance(name_ckpt_mapping, dict)
    for key, val in cluster_dict.items():
        assert key in name_ckpt_mapping
        assert isinstance(val, dict)

    gvr = GridVideoRecorder(
        video_path=video_path,
        env_name=env_name,
        run_name=run_name,
        local_mode=local_mode
    )

    # name_col_mapping = _build_name_col_mapping(cluster_dict)
    # name_row_mapping = _build_name_row_mapping(cluster_dict)
    new_name_ckpt_mapping, name_loc_mapping, name_row_mapping, \
    name_col_mapping = transform_name_ckpt_mapping(
        name_ckpt_mapping, cluster_dict, max_num_cols=max_num_cols
    )

    assert new_name_ckpt_mapping.keys() == name_row_mapping.keys(
    ) == name_loc_mapping.keys()

    frames_dict, extra_info_dict = gvr.generate_frames(
        new_name_ckpt_mapping,
        num_steps=steps,
        seed=seed,
        name_column_mapping=name_col_mapping,
        name_row_mapping=name_row_mapping,
        name_loc_mapping=name_loc_mapping,
        num_workers=num_workers
    )

    gvr.generate_video(frames_dict, extra_info_dict)

    gvr.close()


if __name__ == '__main__':
    import yaml
    import numpy as np

    with open("data/0811-random-test.yaml", 'r') as f:
        name_ckpt_list = yaml.safe_load(f)

    name_ckpt_mapping = OrderedDict()
    for d in name_ckpt_list:
        name_ckpt_mapping[d["name"]] = d["path"]

    fake_cluster_dict = {}
    for i, name in enumerate(name_ckpt_mapping.keys()):
        fake_cluster_dict[name] = {
            "name": name,
            "distance": np.random.random(),
            "cluster": int(i / 2)
        }

    generate_video_of_cluster(
        cluster_dict=fake_cluster_dict,
        name_ckpt_mapping=name_ckpt_mapping,
        video_path="data/0811-random-test",
        env_name="BipedalWalker-v2",
        run_name="PPO",
        seed=0,
        local_mode=False,
        steps=50
    )
