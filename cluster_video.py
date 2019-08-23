# from process_cluster import ClusterFinder
from collections import OrderedDict

from record_video import GridVideoRecorder
from process_data import get_name_ckpt_mapping


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


def _transform_name_ckpt_mapping(
        name_ckpt_mapping, prediction, max_num_cols=10
):
    # Function:
    # 1. re-order the OrderedDict
    # 2. add distance information in name
    clusters = set([c_info['cluster'] for c_info in prediction.values()])

    new_name_ckpt_mapping = OrderedDict()

    old_row_mapping = _build_name_row_mapping(prediction)
    old_col_mapping = _build_name_col_mapping(prediction)

    name_loc_mapping = {}
    name_row_mapping = {} if old_row_mapping is not None else None
    name_col_mapping = {} if old_col_mapping is not None else None

    for row_id, cls in enumerate(clusters):
        within_one_cluster = {
            k: v
            for k, v in prediction.items() if v['cluster'] == cls
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
        prediction,
        env_name,
        run_name,
        num_agents,
        yaml_path,
        video_predix,
        seed=0,
        max_num_cols=8,
        local_mode=False,
        steps=int(1e10),
        num_workers=5
):
    """
    :param prediction: dict
            key: agent name,
            val: cluster_dict={"distance":float, "cluster":int, "name":str}
    :param name_ckpt_mapping:
    :param video_path:
    :param env_name:
    :param run_name:
    :param max_num_cols:
    :param seed:
    :param local_mode:
    :param steps:
    :param num_workers:
    :return:
    """
    name_ckpt_mapping = get_name_ckpt_mapping(yaml_path, num_agents)

    assert isinstance(prediction, dict)
    assert isinstance(name_ckpt_mapping, dict)
    for key, val in prediction.items():
        assert key in name_ckpt_mapping
        assert isinstance(val, dict)

    assert env_name == "BipedalWalker-v2", "We only support BipedalWalker-v2 currently!"

    gvr = GridVideoRecorder(
        video_path=video_predix,
        env_name=env_name,
        run_name=run_name,
        local_mode=local_mode
    )

    new_name_ckpt_mapping, name_loc_mapping, name_row_mapping, \
        name_col_mapping = _transform_name_ckpt_mapping(
            name_ckpt_mapping, prediction, max_num_cols=max_num_cols
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

    fake_prediction = {}
    for i, name in enumerate(name_ckpt_mapping.keys()):
        fake_prediction[name] = {
            "name": name,
            "distance": np.random.random(),
            "cluster": int(i / 2)
        }

    generate_video_of_cluster(
        fake_prediction, "BipedalWalker-v2", "PPO", 2,
        "data/0811-random-test.yaml", "data/0811-random-test-TMP"
    )
