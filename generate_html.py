import yattag

head = \
"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>JUST A TEST TITLE</title>
</head>
<body>
"""

row_template = """
<div class="column">
{content}
<br>

</div>
"""

table_template = """
<table border="1">
  <tr>
      <th>{agent_name}</th>
  </tr>
  <tr>
    <td><img src="{gif_path}"/></td>
  </tr>
  {other_info}
</table>
"""

def build_other_info(dict):
    str_list = []
    for name, val in dict.items():
        str_list.append("<tr><td>{}: {}</td></tr>".format(name, val))
    return "\n".join(str_list)



bottom = \
"""
</body>
</html>
"""

test_name_gif_path_mapping = {
    "agent 1": {
        "gif_path": "data/vis/gif/test-2-agents/3period/sfsad 1.gif",
        "info": {'performance': 100, "length": 101}
    },
    "agent 2": {
        "gif_path": "data/vis/gif/test-2-agents/3period/same file seed 0.gif",
        "info": {'performance': 90, "length": 100, 'unit': 1}
    }
}

"""
- ablated_unit_index: 80
  agent_name: PPO seed=121 rew=299.35
  checkpoint: data/ppo121_ablation_last2_layer/default_policy-default_model-fc2-unit80/checkpoint_782/checkpoint-782
  env_name: BipedalWalker-v2
  episode_length_max: 1035
  episode_length_mean: 913.184
  episode_length_min: 56
  episode_reward_max: 304.5941759342818
  episode_reward_mean: 290.53563121981864
  episode_reward_min: -123.92529782552334
  kl_divergence: 446.1408386230469
  layer_name: default_policy/default_model/fc2
  name: PPO seed=121 rew=290.54 default_policy/default_model/fc2/unit80
  num_rollouts: 500
  path: data/ppo121_ablation_last2_layer/default_policy-default_model-fc2-unit80/checkpoint_782/checkpoint-782
  performance: 290.53563121981864
  run_name: PPO
  unit_name: default_policy/default_model/fc2/unit80
"""

def parse(name_gif_path_mapping):
    table_list = []
    for name, gif_info in name_gif_path_mapping.items():
        gif_path = gif_info['gif_path']
        table = table_template.format(
            agent_name=name, gif_path=gif_path,
            other_info=build_other_info(gif_info['info'])
        )
        table_list.append(table)
    content = "".join(table_list)

    body = row_template.format(content=content)
    html = head + body + bottom
    with open('test.html', 'w') as f:
        f.write(html)

if __name__ == '__main__':
    parse(test_name_gif_path_mapping)
