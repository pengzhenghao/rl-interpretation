# Walkthrough

:fountain_pen: Peng Zhenghao

:spiral_calendar: Aug 23, 2019



## Cheetsheets

### Calling Independent Functions

#### Generate YAML files from a batch of trained agents

```python
from process_data.process_data import generate_yaml
generate_yaml(
  exp_names=["0811-0to50and100to300", "0811-50to100"],
	run_name="PPO",
  output_path="data/300-ppo.yaml"
)
```



#### Read from YAML files

```python
from process_data.process_data import read_yaml, get_name_ckpt_mapping
name_ckpt_mapping = read_yaml(ckpt, number=None, mode='top')
# get_name_ckpt_mapping = read_yaml
# mode = ['top', 'uniform']
# number = The # of agents to choose
```



#### Generate FFT representation

```python
from represent.process_fft import get_fft_representation
data_frame_dict, representation_dict = get_fft_representation(
        name_ckpt_mapping,
        run_name,
        env_name,
        env_maker,
        num_seeds,
        num_rollouts,
        stack=False,
        normalize="range",
        num_workers=10
)
# dict = key: agent name, val: data_frame / representation vector
```



#### Generate dataframe for clustering

```python
from represent.process_fft import parse_representation_dict

cluster_df = parse_representation_dict(
            representation_dict, 
  	        padding="fix",
  			padding_length=500,
  			padding_value=0
)
```



#### Cluster and predict

```python
from cluster.process_cluster import ClusterFinder

cluster_finder = ClusterFinder(
  			cluster_df, 
  			max_num_cluster=None, 
  			standardize=True
)

cluster_finder.display(log=False, save=False, show=True)
# save=False or "PATH_TO_PNG"

best_k = 5 # Input the best k from looking at elbow curve

cluster_finder.set(best_k)

prediction = cluster_finder.predict()
# prediction = 
#			key: agent name, 
# 		val: cluster_dict = {
#					"distance": float, 
#					"cluster": int, 
#					"name": str
#			}
```



#### Generate video from cluster prediction

```python
from cluster.process_cluster import generate_video_of_cluster
generate_video_of_cluster(
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
)
```



### Some Useful APIs

#### From YAML to FFT ClusterFinder

```python
from represent.process_fft import get_fft_cluster_finder

cluster_findr_dict = get_fft_cluster_finder(
        yaml_path,
        env_name,
        env_maker,
        run_name,
  			normalize="range"/"std"/False,
        num_agents=None,
        num_seeds=1,
        num_rollouts=100,
        show=False
)

# cluster_findr_dict = {
#		"nostd_cluster_finder": nostd_cluster_finder,
# 	"std_cluster_finder": std_cluster_finder
# }
```



## Codes Structure

```
.
├── README.md
├── cluster_video.py
├── data
│   ├── 0811-random-test.mp4
│   └── 0811-random-test.yaml
├── format.sh
├── opencv_wrappers.py
├── process_cluster.py
├── process_data.py
├── process_fft.py
├── record_video.py
├── rollout.py
├── startup.py
├── train_PPO.py
└── utils.py
```






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
from process_data.process_data import generate_yaml
generate_yaml(
  exp_names=["0811-0to50and100to300", "0811-50to100"],
	run_name="PPO",
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



## FFT-representation Pipeline

### Step 1

1. Collect Experiences
2. Generate  FFT Representation
3. Store FFT Representation
4. Generate Elbow Curve

```bash
python process_fft.py \
--yaml-path data/300-agents-ppo.yaml \
--num-agents 300
```



### Step 2

You must give the K (`--num-clusters`)

1. Generate Videos
2. Generate Visualization Results

```bash
python process_cluster.py \
--yaml-path data/300-agents-ppo.yaml \
--root data/PREFIX/PREFIX \
--num-agents 300 \
-k 9 \
--num-workers 11 \
--seed 1997
```


