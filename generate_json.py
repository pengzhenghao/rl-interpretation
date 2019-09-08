"""
yaml look like:
- ablated_unit_index: 147
  agent_name: PPO seed=121 rew=299.35
  checkpoint: data/ppo121_ablation/default_policy-default_model-fc_out
  -unit147/checkpoint_782/checkpoint-782
  env_name: BipedalWalker-v2
  episode_length_max: 1038
  episode_length_mean: 888.85
  episode_length_min: 89
  episode_reward_max: 304.0882877733743
  episode_reward_mean: 274.27786567399374
  episode_reward_min: -121.94084630017291
  kl_divergence: 707.9876098632812
  layer_name: default_policy/default_model/fc_out
  name: PPO seed=121 rew=274.28 default_policy/default_model/fc_out/unit147
  num_rollouts: 100
  path: data/ppo121_ablation/default_policy-default_model-fc_out-unit147
  /checkpoint_782/checkpoint-782
  performance: 274.27786567399374
  run_name: PPO
  unit_name: default_policy/default_model/fc_out/unit147

So it is a list of dict.
The json file is used for javascript to parse the gif.
It look like:

{
    "agent 1": {
        "column": 0,
        "row": 1,
        "name": "agent 1",
        "gif_path": {
            "end": "gif/test-2-agents/3period/sfsad 1.gif",
            "clip": "dskljgif/test-2-agents/3period/sfsad 1.gif"
        },
        "info": {
            "performance": 100,
            "length": 101
        }
    },

So its a dict of dict.

This file is used to
    1. generate GIF files for a given experiment.
    2. generate a json file along with those gif for a given experiment.
    3. copy the html file to the exp directory.
"""

import argparse
import json
import os.path as osp

from process_data import read_yaml
from utils import initialize_ray
from record_video import generate_gif

JSON_FILE_NAME = "index.json"
HTML_FILE_NAME = "index.html"
DEFAULT_HTML_FILE_PATH = "data/vis/example.html"
DEFAULT_JS_FILE_PATH = "data/vis/default_layout.js"
DEFAULT_FILES_PATH = {
    "index.html": "data/vis/example_index.html",
    "default_layout.js": "data/vis/default_layout.js",
    "jquery-3.4.1.min.js": "data/vis/jquery-3.4.1.min.js"
}

def generate_json(yaml_path, name_path_dict, output_dir):
    """
    name_path_dict = {
        agent_name: {
            mode: data/vis/exp/agent/gif
        }
    }
    """
    def modify(mode_path_dict):
        return {mode: osp.join(*path.split("/")[3:])
                for mode, path in mode_path_dict.items()}

    name_ckpt_dict = read_yaml(yaml_path)
    json_dict = {}
    for agent_name, info in name_ckpt_dict.items():
        json_dict[agent_name] = {
            "info": info,
            "gif_path": modify(name_path_dict[agent_name]),
            "name": agent_name
        }
    json_path = osp.join(output_dir, JSON_FILE_NAME)
    with open(json_path, 'w') as f:
        json.dump(json_dict, f)
    return json_path


def copy_files(exp_base_dir, file_dict=None):
    file_dict = file_dict or DEFAULT_FILES_PATH
    output_file_dict = {}
    for output_name, file_path in file_dict.items():
        with open(file_path, 'r') as f:
            data = f.read()
        output_path = osp.join(exp_base_dir, output_name)
        with open(output_path, 'w') as f:
            f.write(data)
        output_file_dict[output_name] = output_path
    return output_file_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-path", type=str, required=True)
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--num-gpus", "-g", type=int, default=4)
    args = parser.parse_args()

    initialize_ray(test_mode=args.test_mode, num_gpus=args.num_gpus)

    name_path_dict = generate_gif(args.yaml_path, args.exp_dir)
    print("Finish generate gif.")
    generate_json(args.yaml_path, name_path_dict, args.exp_dir)
    print("Finish generate json.")
    output_file_dict = copy_files(args.exp_dir)
    print("Finish generate html.")
    print("Generate files: ", output_file_dict)
