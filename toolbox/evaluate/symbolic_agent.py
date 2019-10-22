import numpy as np

from toolbox.evaluate.evaluate_utils import restore_agent_with_mask


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
    def __init__(self, ckpt_info, mask_callback_info=None):
        super(MaskSymbolicAgent, self).__init__()
        self.ckpt_info = ckpt_info
        self.agent_info = self.ckpt_info.copy()
        self.agent_info['agent'] = None
        self.agent_info['parent'] = self.ckpt_info['name']
        self.agent_info['mask'] = None

        self.mask = None
        self.mask_callback_info = mask_callback_info

    def mask_callback(self, agent):
        if self.mask is not None:
            agent.set_mask(self.mask)
            return agent
        elif self.mask_callback_info is None:
            # initialize with all ones.
            mask_template = agent.get_mask()
            mask_dict = {
                k: np.ones(shape)
                for k, shape in mask_template.items()
            }
            self.mask = mask_dict
            agent.set_mask(mask_dict)
            return agent
        else:
            return self.add_gaussian_perturbation(
                agent, self.mask_callback_info['mean'],
                self.mask_callback_info['std'], self.mask_callback_info['seed']
            )

    def add_gaussian_perturbation(self, agent, mean, std, seed=None):
        # if self.mask is None:
        assert self.mask is None
        mask_template = agent.get_mask()
        if seed is not None:
            random_state = np.random.RandomState(seed)
        else:
            random_state = np.random
        self.mask = {}
        for mask_name, shape in mask_template.items():
            self.mask[mask_name] = \
                random_state.normal(loc=mean, scale=std,
                                    size=shape)
        agent.set_mask(self.mask)
        return agent

    def clear(self):
        if self.initialized:
            self.agent = None
            self.agent_info['agent'] = None
            # we do not clear the mask, so that the agent is maintained and
            # recoverable! This is really important.

    def get(self, existing_agent=None):
        if self.initialized:
            return self.agent_info
        if not self.initialized and self.mask is not None:
            print(
                "Symbolic Agent is not initialized but the mask exist,"
                "which means it is once initialized but then cleared."
            )
        ckpt = self.ckpt_info
        run_name = ckpt['run_name']
        ckpt_path = ckpt['path']
        env_name = ckpt['env_name']
        self.agent = restore_agent_with_mask(
            run_name, ckpt_path, env_name, existing_agent=existing_agent
        )
        self.agent = self.mask_callback(self.agent)
        self.agent_info['agent'] = self.agent
        self.agent_info['mask'] = self.mask
        return self.agent_info
