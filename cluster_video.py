from process_cluster import ClusterFinder
from record_video import GridVideoRecorder


def _build_name_row_mapping(cluster_dict):
    pass


def _build_name_col_mapping(cluster_dict):
    pass


def _build_name_loc_mapping(cluster_dict):
    pass


def generate_video_of_cluster(
        cluster_dict, name_ckpt_mapping, video_path, env_name, run_name, seed
):

    assert isinstance(cluster_dict, dict)
    assert isinstance(name_ckpt_mapping, dict)
    for key, val in cluster_dict.items():
        assert key in name_ckpt_mapping
        assert isinstance(val, int)

    gvr = GridVideoRecorder(
        video_path=video_path,
        env_name=env_name,
        run_name=run_name,
        # local_mode=args.local_mode
    )

    name_col_mapping = _build_name_col_mapping(cluster_dict)
    name_row_mapping = _build_name_row_mapping(cluster_dict)
    name_loc_mapping = _build_name_loc_mapping(cluster_dict)

    frames_dict, extra_info_dict = gvr.generate_frames(
        name_ckpt_mapping,
        seed=seed,
        name_column_mapping=name_col_mapping,
        name_row_mapping=name_row_mapping,
        name_loc_mapping=name_loc_mapping
    )

    gvr.generate_video(frames_dict, extra_info_dict)

    gvr.close()
