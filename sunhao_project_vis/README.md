# The Visualization Codes for paper "UNDER REVIEW"

## Quick start

```python
from .sunhao_project_vis import *

# Step 1: Load checkpoint and collect frames.
agent_name = "test-sunhao-0924-halfcheetah"
vis_env = "HalfCheetah-v3"
ckpt_dir_path = "Train_PPO_walker/HalfCheetah/CheckPoints"
halfcheetah_result, halfcheetah_result_ppo = collect_frames_from_ckpt_dir(
    ckpt_dir_path, agent_name, vis_env, num_steps=500, reward_threshold=600)


# Step 2: Draw a multiple-exposure figure based on the frames collect.
# In this example, we only draw the "our method" result.
halfcheetah_config = dict(
    start=0,
    interval=300,
    skip_frame=20,
    alpha=0.48,
    velocity_multiplier=7
)
ret, fig_dict = draw_all_result(halfcheetah_result,
                                config=halfcheetah_config)


# Step 3: Concatenate all figures and spread them in a big image.
canvas = draw_multiple_rows(
    ret,
    fig_dict,
    halfcheetah_result,
    choose=None,
    width=5000,
    clip=100,
    put_text=True
)
```

Or you can simply 


The process has three steps:
    1. Load checkpoint, run environment and collect a sequence of frames.
    2. For each agent, generate the multiple-exposure figure.
    3. Concatenate the figure of different agents and add extra
        information like the reward or the agent name (not supported yet).

It should be note that this codes is organized in the very naive way and
along with the future development of our toolbox this code may not
compatible anymore --- That's the reason we left the codes in this branch。

2019.09.26 Peng Zhenghao