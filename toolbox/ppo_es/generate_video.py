import os

from toolbox import initialize_ray
from toolbox.evaluate.rollout import rollout
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.ppo_es.tnb_es import TNBESTrainer
from toolbox.visualize.record_video import GridVideoRecorder

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--novelty-mode", type=str, default="mse")
    args = parser.parse_args()

    initialize_ray(test_mode=False, local_mode=False)

    ckpt = "~/ray_results/1228-tnbes-5agent-walker-large" \
           "/TNBES_MultiAgentEnvWrapper_4_novelty_type=mse,seed=200," \
           "update_steps=0," \
           "use_novelty_value_network=True_2019-12-28_10-48-349e8joanr" \
           "/checkpoint_1645/checkpoint-1645" if args.novelty_mode == "mse" \
        else "~/ray_results/1228-tnbes-5agent-walker-large" \
             "/TNBES_MultiAgentEnvWrapper_5_novelty_type=kl,seed=200," \
             "update_steps=0," \
             "use_novelty_value_network=True_2019-12-28_10-48-35vv14s1ed" \
             "/checkpoint_1600/checkpoint-1600"

    env_name = "Walker2d-v3"
    # env_name = "BipedalWalker-v2"
    num_agents = 5
    fps = 50
    seed = 200
    steps = None

    # wkload = pickle.load(open(os.path.expanduser(ckpt), 'rb'))['worker']
    # states = pickle.loads(wkload)['state']

    for i in range(num_agents):
        agent_id = "agent{}".format(i)

        walker_config = {
            # can change
            "update_steps": 0,
            "use_tnb_plus": False,
            "novelty_type": args.novelty_mode,
            "use_novelty_value_network": True,
            "env": MultiAgentEnvWrapper,
            "env_config": {
                "env_name": env_name,
                "num_agents": num_agents,
                "render_policy": agent_id
            },
            "lr": 0.0,
            "num_gpus": 0,
            'num_workers': 0,
        }
        agent = TNBESTrainer(config=walker_config, env=MultiAgentEnvWrapper)

        agent.restore(os.path.expanduser(ckpt))

        env = MultiAgentEnvWrapper({
            "env_name": env_name,
            "num_agents": num_agents,
            "render_policy": agent_id
        })

        env.seed(seed)

        ret = rollout(agent,
                      env,
                      "ENV_NAME",
                      require_frame=True,
                      require_full_frame=True,
                      num_steps=steps,
                      multiagent_environment=True)
        frames = ret['frames']
        frames = frames[..., 2::-1]
        frames_dict = dict({agent_id: {
            "frames": frames,
            "column": None,
            "row": None,
            "loc": None,
            "period": 0
        }})
        new_extra_info_dict = dict(
            frame_info={
                "width": ret['frames'][0].shape[1],
                "height": ret['frames'][0].shape[0]
            })

        rew = ret['frame_extra_info']['reward'][-1][agent_id]
        print('Accumulated reward for agent <{}>: {}'.format(
            agent_id, rew))

        video_path = "data/1228-tnbes-5agent-walker-large/{}-novelty_mode" \
                     "{}-rew{:.4f}".format(
            agent_id, args.novelty_mode, rew)
        gvr = GridVideoRecorder(video_path, fps=fps, require_full_frame=True)

        path = gvr.generate_single_video(frames_dict)

        print('Agent: <{}> video has been saved at <{}>.'.format(
            agent_id, os.path.abspath(path)))
        # break
