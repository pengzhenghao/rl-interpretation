import numpy as np
from torch import Tensor


class calc_policy_novelty(object):
    # TODO have remove the default thresh.
    # def __init__(self, Policy_Buffer, THRESH=config.thres, dis_type='min'):
    def __init__(self, Policy_Buffer, THRESH, dis_type='min'):
        self.Policy_Buffer = Policy_Buffer
        self.num_of_policies = len(Policy_Buffer)
        self.novelty_recorder = np.zeros(self.num_of_policies)
        self.novelty_recorder_len = 0
        self.THRESH = THRESH
        self.dis_type = dis_type

    def calculate(self, state, action):
        if len(self.Policy_Buffer) == 0:
            return 0
        for i, key_i in enumerate(self.Policy_Buffer.keys()):
            self.Policy_Buffer[key_i].eval()
            a_mean, a_logstd, val, val_novel = self.Policy_Buffer[
                key_i].forward((Tensor(state).float().unsqueeze(0).cuda()))
            self.novelty_recorder[i] += np.linalg.norm(
                a_mean.cpu().detach().numpy() - action.cpu().detach().numpy()
            )

        self.novelty_recorder_len += 1
        if self.dis_type == 'min':
            min_novel = np.min(
                self.novelty_recorder / self.novelty_recorder_len
            )
            return min_novel - self.THRESH
        elif self.dis_type == 'max':
            max_novel = np.max(
                self.novelty_recorder / self.novelty_recorder_len
            )
            return max_novel - self.THRESH
