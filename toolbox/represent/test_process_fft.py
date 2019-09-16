from toolbox.process_data import read_batch_yaml, save_yaml
from toolbox.represent.process_fft import get_fft_cluster_finder
from toolbox.utils import initialize_ray


def test_get_fft_cluster_finder():
    num_rollouts = 5
    num_workers = 10
    yaml_path_dict_list = [
        {
            "number": 2,
            "path": "data/yaml/ppo-300-agents.yaml",
        }, {
            "number": 2,
            "path": "data/es-30-agents-0818.yaml"
        }
    ]

    yaml_output_path = "delete_me_please.yaml"

    name_ckpt_mapping = read_batch_yaml(yaml_path_dict_list)
    yaml_output_path = save_yaml(name_ckpt_mapping, yaml_output_path)

    ret = get_fft_cluster_finder(
        yaml_path=yaml_output_path,
        num_rollouts=num_rollouts,
        num_workers=num_workers,
        num_gpus=3
    )
    print(ret)
    return ret


if __name__ == '__main__':
    import os

    initialize_ray(test_mode=True, num_gpus=3)

    os.chdir("../../")
    ret = test_get_fft_cluster_finder()
