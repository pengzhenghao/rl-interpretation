from toolbox.evaluate.evaluate_utils import restore_agent_with_mask
from toolbox.utils import get_random_string

import numpy as np

class SymbolicAgentBase:

    def __init__(self):
        self.agent = None
        self.agent_info = {}

    @property
    def initialized(self):
        return self.agent is not None

    def get(self):
        raise NotImplementedError


class MaskSymbolicAgent(SymbolicAgentBase):

    def __init__(self, ckpt_info, mask_callback=None):
        super(MaskSymbolicAgent, self).__init__()
        self.ckpt_info = ckpt_info
        self.mask_callback = mask_callback

    def get(self):
        if self.initialized:
            return self.agent
        ckpt = self.ckpt_info
        run_name = ckpt['run_name']
        ckpt_path = ckpt['path']
        env_name = ckpt['env_name']
        self.agent = restore_agent_with_mask(run_name, ckpt_path, env_name)




        mask_template = self.agent.get_mask_info()
        mask_dict = {k: np.ones((shape[1],)) for k, shape in
                     mask_template.items()}
        self.agent.get_policy().set_default_mask(
            mask_dict
        )

        if self.mask_callback is not None:
            assert callable(self.mask_callback)
            self.agent = self.mask_callback(self.agent)

        self.agent_info.update(ckpt)
        self.agent_info['agent'] = self.agent
        self.agent_info['id'] = 0
        self.agent_info['parent'] = self.ckpt_info['name']
        self.agent_info['mask'] = mask_dict

        return self.agent


def add_gaussian_perturbation(agent, mean, std, seed=None):
    mask_template = agent.get_mask_info()
    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random
    mask = {}
    for mask_name, shape in mask_template.items():
        mask[mask_name] = \
            random_state.normal(loc=mean, scale=std,
                                size=shape[1:])
    agent.get_policy().set_default_mask(mask)
    return agent
