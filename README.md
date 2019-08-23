# Walkthrough

:fountain_pen: Peng Zhenghao

:spiral_calendar: Aug 22, 2019



## Generate YAML files to conclude batch training

We have build the YAML files which summary the training of a large batch of agents:

```yaml
- name: PPO seed=139 rew=220.58
  path: /XXX/ray_results/0811-0to50and100to300/PPO_BipedalWalker-v2_39_seed=139_2019-08-11_23-00-21q3yaz5c6/checkpoint_782/checkpoint-782
  performance: 220.57752118040958
- name: PPO seed=288 rew=224.95
  path: /XXX/ray_results/0811-0to50and100to300/PPO_BipedalWalker-v2_188_seed=288_2019-08-12_11-50-56guywuhds/checkpoint_782/checkpoint-782
  performance: 224.94733969119042
```

After convert to the python object by running `read_yaml` at `process_data`:

```python
def read_yaml(ckpt):
    with open(ckpt, 'r') as f:
        name_ckpt_list = yaml.safe_load(f)
    name_ckpt_mapping = OrderedDict()
    for d in name_ckpt_list:
        name_ckpt_mapping[d["name"]] = d["path"]
    return name_ckpt_mapping
```

We get a OrderedDict.



So above is the format of yaml files which give an OrderedDict with name of agent as key, and the checkpoint path / performance as the values.

How we generate it? Just call:

```python
from process_data import generate_yaml
generate_yaml(
  exp_names=["0811-0to50and100to300", "0811-50to100"],
	algo_name="PPO",
  output_path="data/300-ppo.yaml"
)
```

Then the perfect yaml file will be stored at your `output_path`.

Or you can simply use command line:

```bash
python process_data.py \
--exp-names 0811-0to50and100to300 0811-50to100 \
--algo-name PPO \
--output-path data/300-ppo.yaml
```



