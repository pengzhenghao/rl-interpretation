import logging

import numpy as np
import copy
from toolbox.evaluate.evaluate_utils import restore_agent_with_mask

logger = logging.getLogger(__name__)


class SymbolicAgentBase(object):
    def __init__(self):
        self.agent = None
        self.agent_info = {}
        self.name = "SymbolicAgentBase"

    @property
    def initialized(self):
        return self.agent is not None

    def get(self):
        raise NotImplementedError

    def clear(self):
        pass


class MaskSymbolicAgent(SymbolicAgentBase):
    def __init__(self, ckpt_info, mask_callback_info=None, name=None, existing_weights=None):
        super(MaskSymbolicAgent, self).__init__()
        self.ckpt_info = ckpt_info
        self.agent_info = self.ckpt_info.copy()
        self.agent_info['agent'] = None
        self.agent_info['parent'] = self.ckpt_info['name']
        self.agent_info['mask'] = None

        if name is not None:
            self.name = name
        else:
            self.name = self.agent_info['name']

        self.mask = None
        self.mask_callback_info = mask_callback_info
        self.weights = existing_weights

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
            self.agent_info.pop('agent')
            del self.agent
            self.agent = None
            self.agent_info['agent'] = None
            # we do not clear the mask, so that the agent is maintained and
            # recoverable! This is really important.

    def get(self, existing_agent=None, existing_weights=None):
        if self.initialized:
            return self.agent_info
        if not self.initialized and self.mask is not None:
            logger.info(
                "Symbolic Agent is not initialized but the mask exist,"
                "which means it is once initialized but then cleared."
            )
        ckpt = self.ckpt_info
        run_name = ckpt['run_name']
        ckpt_path = ckpt['path']
        env_name = ckpt['env_name']

        if existing_weights is not None:
            self.weights = existing_weights.copy()
            ckpt_path = None
            logger.info("Override the ckpt with you provided agent weights!")

        self.agent = restore_agent_with_mask(
            run_name, ckpt_path, env_name, existing_agent=existing_agent
        )
        self.agent = self.mask_callback(self.agent)

        if self.weights is not None:
            self.agent.set_weights(self.weights)
            logger.info("Successfully set weights for agent.")

        self.weights = self.agent.get_weights()
        self.agent_info['agent'] = self.agent
        self.agent_info['mask'] = self.mask
        self.agent_info['weights'] = self.weights
        return self.agent_info

    def __getstate__(self):
        if self.initialized:
            logger.warning(
                "So Terrible! The SymbolicAgent <{}> is not cleared! I clear "
                "it for you.".format(self.name)
            )
            self.clear()
        assert not self.initialized
        return copy.deepcopy(self.__dict__)
