import os.path as osp

from toolbox.abstract_worker import WorkerManagerBase, WorkerBase
from toolbox.visualize.record_video import GridVideoRecorder


def generate_trailer_from_agent(
        agent, agent_name, output_path, require_full_frame=False, _steps=None
):
    env_name = agent.config['env']
    fps = 50 if env_name.startswith("BipedalWalker") else 20

    gvr = GridVideoRecorder(
        video_path=output_path, fps=fps, require_full_frame=require_full_frame
    )
    frames_dict, extra_info_dict = gvr.generate_frames_from_agent(
        agent, agent_name, num_steps=_steps or 1000, ideal_steps=200
    )

    path = gvr.generate_single_video(frames_dict)
    return path


class _SymbolicAgentVideoWorker(WorkerBase):
    def __init__(self):
        # We don't reuse agent here, because it lead to strange bug..
        # and the video-generation is time consuming
        # so it's not necessary to reuse agent.
        self.existing_agent = None

    def run(self, symbolic_agent, agent_name, output_path):
        agent = symbolic_agent.get()['agent']
        path = generate_trailer_from_agent(agent, agent_name, output_path)
        return path


class RemoteSymbolicAgentVideoManager(WorkerManagerBase):
    def __init__(self, num_workers, total_num=None, log_interval=1):
        super(RemoteSymbolicAgentVideoManager, self).__init__(
            num_workers, _SymbolicAgentVideoWorker, total_num, log_interval,
            "generate video"
        )

    def generate_video(self, agent_name, symbolic_agent, base_output_path):
        agent_name = agent_name.replace(' ', '_')
        output_path = osp.join(base_output_path, agent_name)
        self.submit(agent_name, symbolic_agent, agent_name, output_path)

    def parse_result(self, result):
        """result is path here."""
        return "Saved at: {}".format(result)


def test():
    from collections import OrderedDict
    from toolbox.evaluate import MaskSymbolicAgent
    from toolbox.utils import initialize_ray, get_random_string
    import shutil

    initialize_ray(test_mode=True)

    print("Finish init")

    num_workers = 4
    num_agents = 8
    base_output_path = "/tmp/generate_trailer"
    ckpt = {
        "path":
        "~/ray_results/0810-20seeds/PPO_BipedalWalker-v2_"
        "0_seed=20_2019-08-10_16-54-37xaa2muqm/"
        "checkpoint_469/checkpoint-469",
        "env_name":
        "BipedalWalker-v2",
        "run_name":
        "PPO"
    }

    shutil.rmtree(base_output_path, ignore_errors=True)

    master_agents = OrderedDict()

    for _ in range(num_agents):
        ckpt['name'] = get_random_string()
        agent = MaskSymbolicAgent(ckpt)
        master_agents[ckpt['name']] = agent

    print("Master agents: ", master_agents)

    rsavm = RemoteSymbolicAgentVideoManager(num_workers, len(master_agents))

    for name, symbolic_agent in master_agents.items():
        rsavm.generate_video(name, symbolic_agent, base_output_path)

    result = rsavm.get_result()
    print(result)

    return result


if __name__ == '__main__':
    ret = test()
