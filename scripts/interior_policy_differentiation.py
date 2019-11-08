import numpy as np
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import compute_advantages, SampleBatch, \
    PPOTFPolicy

from toolbox import initialize_ray
import copy

IPDPPO_default_config = copy.deepcopy(DEFAULT_CONFIG)
IPDPPO_default_config.update(
    ADDYOURCONFIG=None
)

def postprocess_ppo_gae_modified(policy,
                                 sample_batch,
                                 other_agent_batches=None,
                                 episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    print("YOU HAVE ENTER HERE.")

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch


IPDPPOTFPolicy = PPOTFPolicy.with_updates(
    name="IPDPPOTFPolicy",
    postprocess_fn=postprocess_ppo_gae_modified
)

IPDPPOTrainer = PPOTrainer.with_updates(
    name="IPDPPO",
    default_config=IPDPPO_default_config,
    default_policy=IPDPPOTFPolicy,
)


def _build_matrix(iterable, apply_function, default_value=0):
    """
    Copied from toolbox.interface.cross_agent_analysis
    """
    length = len(iterable)
    matrix = np.empty((length, length))
    matrix.fill(default_value)
    for i1 in range(length - 1):
        for i2 in range(i1, length):
            repr1 = iterable[i1]
            repr2 = iterable[i2]
            result = apply_function(repr1, repr2)
            matrix[i1, i2] = result
            matrix[i2, i1] = result
    return matrix


def test_maddpg_custom_metrics():
    # Comments on this function may be meaningless...
    def on_episode_start(info):
        print("Enter on_episode_start, current ep id: {},"
              " ep length: {}, total reward: {}".format(
            info['episode'].episode_id,
            info['episode'].length,
            info['episode'].total_reward,
        ))

    def on_episode_step(info):
        print("Enter on_episode_step, current ep id: {},"
              " ep length: {}, total reward: {}".format(
            info['episode'].episode_id,
            info['episode'].length,
            info['episode'].total_reward,
        ))

        base_env = info['env']

        print(1)

        # for env_id in range(len(base_env.envs)):
        # base_env.try_reset(env_id)
        # print("Finish 'try_reset' for env: ", env_id)

    def on_episode_end(info):
        print("Enter on_episode_end, current ep id: {},"
              " ep length: {}, total reward: {}".format(
            info['episode'].episode_id,
            info['episode'].length,
            info['episode'].total_reward,
        ))

    def on_sample_end(info):
        """The info here contain two things, the worker
        and the samples. The samples is a 'MultiAgentBatch'
        """
        pass

    def on_postprocess_traj(info):
        episode = info["episode"]
        batch = info["post_batch"]
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1

    extra_config = {
        "env": "BipedalWalker-v2",
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
            "on_sample_end": on_sample_end,
            "on_postprocess_traj": on_postprocess_traj,
        },
    }

    initialize_ray(test_mode=True, local_mode=True)
    tune.run(
        IPDPPOTrainer,
        stop={
            "timesteps_total": 1000,
        },
        config=extra_config
    )


if __name__ == "__main__":
    test_maddpg_custom_metrics()

# """This file implement the interior policy differentiation algorithm"""
# import ray
# import numpy as np
#
# class calc_policy_novelty(object):
#     def __init__(self, Policy_Buffer, THRESH=config.thres, dis_type='min'):
#         self.Policy_Buffer = Policy_Buffer
#         self.num_of_policies = len(Policy_Buffer)
#         self.novelty_recorder = np.zeros(self.num_of_policies)
#         self.novelty_recorder_len = 0
#         self.THRESH = THRESH
#         self.dis_type = dis_type
#
#     def calculate(self, state, action):
#         if len(self.Policy_Buffer) == 0:
#             return 0
#         for i, key_i in enumerate(self.Policy_Buffer.keys()):
#             self.Policy_Buffer[key_i].eval()
#             a_mean, a_logstd, val = self.Policy_Buffer[key_i].forward(
#                 (Tensor(state).float().unsqueeze(0).cuda()))
#             self.novelty_recorder[i] += np.linalg.norm(
#                 a_mean.cpu().detach().numpy() - action.cpu().detach(
#                 ).numpy())
#
#         self.novelty_recorder_len += 1
#         if self.dis_type == 'min':
#             min_novel = np.min(
#                 self.novelty_recorder / self.novelty_recorder_len)
#             return min_novel - self.THRESH
#         elif self.dis_type == 'max':
#             max_novel = np.max(
#                 self.novelty_recorder / self.novelty_recorder_len)
#             return max_novel - self.THRESH
#
#
#
#
#     class args(object):
#         env_name = ENV_NAME
#         seed = 1234
#         num_episode = config.num_episode
#         batch_size = 2048
#         max_step_per_round = 2000
#         gamma = 0.995
#         lamda = 0.97
#         log_num_episode = 1
#         num_epoch = 10
#         minibatch_size = 256
#         clip = 0.2
#         loss_coeff_value = 0.5
#         loss_coeff_entropy = 0.01
#         lr = 3e-4
#         num_parallel_run = 1
#         # tricks
#         schedule_adam = 'linear'
#         schedule_clip = 'linear'
#         layer_norm = True
#         state_norm = False
#         advantage_norm = True
#         lossvalue_norm = True
#
#
#     class RunningStat(object):
#         def __init__(self, shape):
#             self._n = 0
#             self._M = np.zeros(shape)
#             self._S = np.zeros(shape)
#
#         def push(self, x):
#             x = np.asarray(x)
#             assert x.shape == self._M.shape
#             self._n += 1
#             if self._n == 1:
#                 self._M[...] = x
#             else:
#                 oldM = self._M.copy()
#                 self._M[...] = oldM + (x - oldM) / self._n
#                 self._S[...] = self._S + (x - oldM) * (x - self._M)
#
#         @property
#         def n(self):
#             return self._n
#
#         @property
#         def mean(self):
#             return self._M
#
#         @property
#         def var(self):
#             return self._S / (self._n - 1) if self._n > 1 else np.square(
#                 self._M)
#
#         @property
#         def std(self):
#             return np.sqrt(self.var)
#
#         @property
#         def shape(self):
#             return self._M.shape
#
#
#     def ppo(args):
#         env = gym.make(args.env_name)
#         num_inputs = env.observation_space.shape[0]
#         num_actions = env.action_space.shape[0]
#
#         env.seed(args.seed)
#         torch.manual_seed(args.seed)
#
#         # network = ActorCritic(num_inputs, num_actions,
#         # layer_norm=args.layer_norm)
#         optimizer = opt.Adam(network.parameters(), lr=args.lr)
#
#         running_state = ZFilter((num_inputs,), clip=5.0)
#
#         # record average 1-round cumulative reward in every episode
#         reward_record = []
#         global_steps = 0
#
#         lr_now = args.lr
#         clip_now = args.clip
#         Best_performance = 0
#         for i_episode in range(args.num_episode):
#             # step1: perform current policy to collect trajectories
#             # this is an on-policy method!
#             memory = Memory()
#
#             num_steps = 0
#             reward_list = []
#             len_list = []
#             performance = 0
#             total_done = 0
#             early_done = 0
#             while num_steps < args.batch_size:
#                 state = env.reset()
#
#                 cpn = calc_policy_novelty(Policy_Buffer=policy_buffer)
#
#                 if args.state_norm:
#                     state = running_state(state)
#                 reward_sum = 0
#                 reward_novel_sum = 0
#                 for t in range(args.max_step_per_round):
#                     action_mean, action_logstd, value = network(
#                         Tensor(state).float().unsqueeze(0).cuda())
#
#                     action, logproba = network.select_action(action_mean,
#                                                              action_logstd)
#                     action = action.cpu().data.numpy()[0]
#                     logproba = logproba.cpu().data.numpy()[0]
#
#                     next_state, reward, done, _ = env.step(action)
#                     # reward_novel = calc_distance(state,action_mean,
#                     # policy_buffer)
#                     reward_novel = cpn.calculate(state, action_mean)
#                     if t >= T_start:
#                         reward_novel_sum += reward_novel
#                         if reward_novel_sum <= Lower_Novel_Bound:  #
#                             # /cpn.novelty_recorder_len[0]:
#                             early_done += 1
#                             done = True
#
#                     if False:  # (i_episode+1)%5 == 0 and num_steps<50:# and
#                         # i_episode>1000:
#                         # time.sleep(1)
#                         show_state(env, t, i_episode)
#                         print('mean action', action_mean)
#                         print('std action', action_logstd)
#                         print('action', action)
#                         print('reward_sum', reward_sum)
#
#                     reward_sum += reward
#                     if args.state_norm:
#                         next_state = running_state(next_state)
#                     mask = 0 if done else 1
#
#                     memory.push(state, value, action, logproba, mask,
#                                 next_state, reward)
#
#                     if done:
#                         total_done += 1
#                         break
#
#                     state = next_state
#
#                 num_steps += (t + 1)
#                 global_steps += (t + 1)
#                 reward_list.append(reward_sum)
#                 len_list.append(t + 1)
#             reward_record.append({
#                 'episode': i_episode,
#                 'steps': global_steps,
#                 'meanepreward': np.mean(reward_list),
#                 'meaneplen': np.mean(len_list)})
#             rwds.extend(reward_list)
#             batch = memory.sample()
#             batch_size = len(memory)
#
#             # step2: extract variables from trajectories
#             rewards = Tensor(batch.reward)
#             values = Tensor(batch.value)
#             masks = Tensor(batch.mask)
#             actions = Tensor(batch.action)
#             states = Tensor(batch.state)
#             oldlogproba = Tensor(batch.logproba)
#
#             returns = Tensor(batch_size)
#             deltas = Tensor(batch_size)
#             advantages = Tensor(batch_size)
#
#             prev_return = 0
#             prev_value = 0
#             prev_advantage = 0
#             for i in reversed(range(batch_size)):
#                 returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
#                 deltas[i] = rewards[i] + args.gamma * prev_value * masks[
#                 i] - \
#                             values[i]
#                 # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization
#                 # advantage estimate)
#                 advantages[i] = deltas[
#                                     i] + args.gamma * args.lamda * \
#                                 prev_advantage * \
#                                 masks[i]
#
#                 prev_return = returns[i]
#                 prev_value = values[i]
#                 prev_advantage = advantages[i]
#             if args.advantage_norm:
#                 advantages = (advantages - advantages.mean()) / (
#                         advantages.std() + EPS)
#
#             for i_epoch in range(
#                     int(args.num_epoch * batch_size / args.minibatch_size)):
#                 # sample from current batch
#                 minibatch_ind = np.random.choice(batch_size,
#                                                  args.minibatch_size,
#                                                  replace=False)
#                 minibatch_states = states[minibatch_ind]
#                 minibatch_actions = actions[minibatch_ind]
#                 minibatch_oldlogproba = oldlogproba[minibatch_ind]
#                 minibatch_newlogproba = network.get_logproba(
#                     minibatch_states.cuda(), minibatch_actions.cuda()).cpu()
#                 minibatch_advantages = advantages[minibatch_ind]
#                 minibatch_returns = returns[minibatch_ind]
#                 minibatch_newvalues = network._forward_critic(
#                     minibatch_states.cuda()).cpu().flatten()
#
#                 ratio = torch.exp(
#                     minibatch_newlogproba - minibatch_oldlogproba)
#                 surr1 = ratio * minibatch_advantages
#                 surr2 = ratio.clamp(1 - clip_now,
#                                     1 + clip_now) * minibatch_advantages
#                 loss_surr = - torch.mean(torch.min(surr1, surr2))
#
#                 # not sure the value loss should be clipped as well
#                 # clip example:
#                 # https://github.com/Jiankai-Sun/Proximal-Policy
#                 # -Optimization-in-Pytorch/blob/master/ppo.py
#                 # however, it does not make sense to clip score-like value
#                 # by a dimensionless clipping parameter
#                 # moreover, original paper does not mention clipped value
#                 if args.lossvalue_norm:
#                     minibatch_return_6std = 6 * minibatch_returns.std()
#                     loss_value = torch.mean(
#                         (minibatch_newvalues - minibatch_returns).pow(
#                             2)) / minibatch_return_6std
#                 else:
#                     loss_value = torch.mean(
#                         (minibatch_newvalues - minibatch_returns).pow(2))
#
#                 loss_entropy = torch.mean(
#                     torch.exp(minibatch_newlogproba) * minibatch_newlogproba)
#
#                 total_loss = loss_surr + args.loss_coeff_value * loss_value \
#                              + args.loss_coeff_entropy * loss_entropy
#                 optimizer.zero_grad()
#                 total_loss.backward()
#                 optimizer.step()
#
#             if args.schedule_clip == 'linear':
#                 ep_ratio = 1 - (i_episode / args.num_episode)
#                 clip_now = args.clip * ep_ratio
#
#             if args.schedule_adam == 'linear':
#                 ep_ratio = 1 - (i_episode / args.num_episode)
#                 lr_now = args.lr * ep_ratio
#                 # set learning rate
#                 # ref: https://stackoverflow.com/questions/48324152/
#                 for g in optimizer.param_groups:
#                     g['lr'] = lr_now
#
#             if i_episode % args.log_num_episode == 0:
#                 print(
#                     'Finished episode: {} Reward: {:.4f} total_loss = {
#                     :.4f} '
#                     '= {:.4f} + {} * {:.4f} + {} * {:.4f}' \
#                         .format(i_episode, reward_record[-1]['meanepreward'],
#                                 total_loss.data, loss_surr.data,
#                                 args.loss_coeff_value,
#                                 loss_value.data, args.loss_coeff_entropy,
#                                 loss_entropy.data))
#                 print('-----------------')
#                 performance = reward_record[-1]['meanepreward'] * (
#                         1 - early_done / total_done)
#                 if performance >= Best_performance:
#                     torch.save(network.state_dict(), ENV_NAME.split('-')[
#                         0] + '/CheckPoints/EarlyStopPolicy_Suc_{0}hidden_{'
#                              '1}threshold_{2}repeat'.format(
#                         config.hid_num, config.thres,
#                         str(repeat) + '_temp_best'))
#                     Best_performance = performance
#
#                     if early_done / total_done < 0.1 and reward_record[-1][
#                         'meanepreward'] > np.mean(final_100_reward):
#                         print('Find Better Novel Policy!')
#                         # policy_buffer[str(repeat)+'_'+str(i_episode)] =
#                         # network
#
#                         torch.save(network.state_dict(), ENV_NAME.split('-')[
#                             0] + '/CheckPoints/EarlyStopPolicy_Suc_{'
#                                  '0}hidden_{1}threshold_{2}repeat'.format(
#                             config.hid_num, config.thres,
#                             str(repeat) + '_' + str(i_episode)))
#                         # break
#                 print('early stop proportion:', early_done / total_done,
#                       'Temp Best Performance:', Best_performance)
#                 print('===============================')
#         return reward_record
#
