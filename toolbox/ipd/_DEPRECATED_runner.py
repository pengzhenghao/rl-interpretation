import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
from torch import Tensor

from toolbox.ipd.calc_reward import calc_policy_novelty
from toolbox.ipd.model import ZFilter, Memory

EPS = 1e-10
T_start = 20
Lower_Novel_Bound = -0.1


def ppo(args, network, policy_buffer, config):
    # TODO add network as pass in

    ENV_NAME = args.env_name
    env = gym.make(args.env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    rwds = []

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    # network = ActorCritic(num_inputs, num_actions,
    # layer_norm=args.layer_norm)
    optimizer = opt.Adam(network.parameters(), lr=args.lr)

    running_state = ZFilter((num_inputs, ), clip=5.0)

    # record average 1-round cumulative reward in every episode
    reward_record = []
    reward_record_novel = []
    global_steps = 0

    lr_now = args.lr
    clip_now = args.clip
    Best_performance = 0

    for i_episode in range(args.num_episode):
        # step1: perform current policy to collect trajectories
        # this is an on-policy method!
        memory = Memory()

        num_steps = 0
        reward_list = []
        reward_list_novel = []
        len_list = []
        performance = 0
        total_done = 0
        early_done = 0
        reward_novel_sum = 0
        while num_steps < args.batch_size:
            state = env.reset()

            cpn = calc_policy_novelty(Policy_Buffer=policy_buffer)

            if args.state_norm:
                state = running_state(state)
            reward_sum = 0
            # reward_novel_sum = 0

            for t in range(args.max_step_per_round):
                action_mean, action_logstd, value, choreo_value = network(
                    Tensor(state).float().unsqueeze(0).cuda()
                )

                action, logproba = network.select_action(
                    action_mean, action_logstd
                )
                action = action.cpu().data.numpy()[0]
                logproba = logproba.cpu().data.numpy()[0]

                next_state, reward, done, _ = env.step(action)
                # reward_novel = calc_distance(state,action_mean,
                # policy_buffer)
                reward_novel = cpn.calculate(state, action_mean)
                if t >= T_start:
                    reward_novel_sum += reward_novel
                '''
                TNB
                '''
                # reward_novel_sum += reward_novel
                reward_sum += reward

                if args.state_norm:
                    next_state = running_state(next_state)
                mask = 0 if done else 1

                memory.push(
                    state, value, choreo_value, action, logproba, mask,
                    next_state, reward, reward_novel
                )

                if done:
                    total_done += 1
                    break

                state = next_state

            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_list.append(reward_sum)
            reward_list_novel.append(reward_novel_sum)
            len_list.append(t + 1)

        reward_record.append(
            {
                'episode': i_episode,
                'steps': global_steps,
                'meanepreward': np.mean(reward_list),
                'meaneplen': np.mean(len_list)
            }
        )
        reward_record_novel.append(
            {
                'episode': i_episode,
                'steps': global_steps,
                'meanepreward': np.mean(reward_list_novel),
                'meaneplen': np.mean(len_list)
            }
        )

        rwds.extend(reward_list)
        batch = memory.sample()
        batch_size = len(memory)

        # step2: extract variables from trajectories
        rewards = Tensor(batch.reward)
        values = Tensor(batch.value)
        masks = Tensor(batch.mask)
        actions = Tensor(batch.action)
        states = Tensor(batch.state)
        oldlogproba = Tensor(batch.logproba)

        returns = Tensor(batch_size)
        deltas = Tensor(batch_size)
        advantages = Tensor(batch_size)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        '''----Start for Choreo----'''
        rewards_novel = Tensor(batch.reward_novel)
        values_novel = Tensor(batch.choreo_value)
        masks_novel = masks
        actions_novel = actions
        states_novel = states
        oldlogproba_novel = oldlogproba

        returns_novel = Tensor(batch_size)
        deltas_novel = Tensor(batch_size)
        advantages_novel = Tensor(batch_size)

        prev_return_novel = 0
        prev_value_novel = 0
        prev_advantage_novel = 0
        '''----End for Choreo----'''

        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - \
                        values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization
            # advantage estimate)
            advantages[i] = deltas[i] + args.gamma * args.lamda * \
                            prev_advantage * \
                            masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
            '''Start Choreo'''
            returns_novel[i] = rewards_novel[
                                   i] + args.gamma * prev_return_novel * \
                               masks_novel[i]
            deltas_novel[i] = rewards_novel[
                                  i] + args.gamma * prev_value_novel * \
                              masks_novel[i] - values_novel[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization
            # advantage estimate)
            advantages_novel[i] = deltas_novel[
                                      i] + args.gamma * args.lamda * \
                                  prev_advantage_novel * \
                                  masks_novel[i]

            prev_return_novel = returns_novel[i]
            prev_value_novel = values_novel[i]
            prev_advantage_novel = advantages_novel[i]
            '''End Choreo'''

        if args.advantage_norm:
            advantages = (advantages -
                          advantages.mean()) / (advantages.std() + EPS)
            '''Start Choreo'''
            advantages_novel = (advantages_novel - advantages_novel.mean()
                                ) / (advantages_novel.std() + EPS)
            '''End Choreo'''

        for i_epoch in range(int(args.num_epoch * batch_size /
                                 args.minibatch_size)):
            # sample from current batch
            minibatch_ind = np.random.choice(
                batch_size, args.minibatch_size, replace=False
            )
            minibatch_states = states[minibatch_ind]
            minibatch_actions = actions[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            minibatch_newlogproba = network.get_logproba(
                minibatch_states.cuda(), minibatch_actions.cuda()
            ).cpu()
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_newvalues = network._forward_critic(
                minibatch_states.cuda()
            ).cpu().flatten()

            ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(
                1 - clip_now, 1 + clip_now
            ) * minibatch_advantages
            loss_surr = -torch.mean(torch.min(surr1, surr2))

            # not sure the value loss should be clipped as well
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy
            # -Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a
            # dimensionless clipping parameter
            # moreover, original paper does not mention clipped value
            if args.lossvalue_norm:
                minibatch_return_6std = 6 * minibatch_returns.std()
                loss_value = torch.mean(
                    (minibatch_newvalues - minibatch_returns).pow(2)
                ) / minibatch_return_6std
            else:
                loss_value = torch.mean(
                    (minibatch_newvalues - minibatch_returns).pow(2)
                )

            loss_entropy = torch.mean(
                torch.exp(minibatch_newlogproba) * minibatch_newlogproba
            )

            total_loss = loss_surr + args.loss_coeff_value * loss_value + \
                         args.loss_coeff_entropy * loss_entropy
            '''Start Choreo'''
            minibatch_states_novel = states_novel[minibatch_ind]
            minibatch_actions_novel = actions_novel[minibatch_ind]
            minibatch_oldlogproba_novel = oldlogproba_novel[minibatch_ind]
            minibatch_newlogproba_novel = network.get_logproba(
                minibatch_states_novel.cuda(), minibatch_actions_novel.cuda()
            ).cpu()
            minibatch_advantages_novel = advantages_novel[minibatch_ind]
            minibatch_returns_novel = returns_novel[minibatch_ind]
            minibatch_newvalues_novel = network._forward_choreo(
                minibatch_states_novel.cuda()
            ).cpu().flatten()

            ratio_novel = torch.exp(
                minibatch_newlogproba_novel - minibatch_oldlogproba_novel
            )
            surr1_novel = ratio_novel * minibatch_advantages_novel
            surr2_novel = ratio_novel.clamp(1 - clip_now,
                                            1 + clip_now) * \
                          minibatch_advantages_novel
            loss_surr_novel = -torch.mean(torch.min(surr1_novel, surr2_novel))

            # not sure the value loss should be clipped as well
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy
            # -Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a
            # dimensionless clipping parameter
            # moreover, original paper does not mention clipped value
            if args.lossvalue_norm:
                minibatch_return_6std_novel = 6 * \
                                              minibatch_returns_novel.std() \
                                              + EPS
                loss_value_novel = torch.mean(
                    (minibatch_newvalues_novel -
                     minibatch_returns_novel).pow(2)
                ) / minibatch_return_6std_novel
            else:
                loss_value_novel = torch.mean(
                    (minibatch_newvalues_novel -
                     minibatch_returns_novel).pow(2)
                )

            loss_entropy_novel = torch.mean(
                torch.exp(minibatch_newlogproba_novel) *
                minibatch_newlogproba_novel
            )

            total_loss_novel = loss_surr_novel + args.loss_coeff_value * \
                               loss_value_novel + args.loss_coeff_entropy * \
                               loss_entropy_novel
            '''End Choreo'''

            if len(policy_buffer) == 0:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            else:  # by Guodong Xu
                if reward_novel_sum >= Lower_Novel_Bound:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                else:
                    '''Start Choreo'''
                    optimizer.zero_grad()
                    total_loss_novel.backward(retain_graph=True)  # novel loss
                    grad1 = []
                    for param in network.parameters():
                        try:
                            grad1.append(param.grad.data.flatten())
                        except:
                            grad1.append(torch.zeros_like(param).flatten())
                    grad1 = torch.cat(grad1)
                    # second grad
                    optimizer.zero_grad()
                    total_loss.backward()
                    grad2 = []
                    for param in network.parameters():
                        try:
                            grad2.append(param.grad.data.flatten())
                        except:
                            grad2.append(torch.zeros_like(param).flatten())
                    grad2 = torch.cat(grad2)
                    # compute grad
                    cos = F.cosine_similarity(grad1, grad2, dim=0)
                    norm1 = torch.norm(grad1, p=2)
                    norm2 = torch.norm(grad2, p=2)
                    if cos > 0:  # less than 90
                        grad = grad1 / norm1 + grad2 / norm2
                    else:
                        grad = -(cos / norm1) * grad1 + (1 / norm2) * grad2
                    grad = grad / torch.norm(grad, p=2) * (norm1 + norm2) / 2
                    # fill the grad into param.grad
                    base = 0
                    optimizer.zero_grad()
                    for param in network.parameters():
                        param.grad += grad[base:base +
                                           param.numel()].reshape_as(param)
                        base += param.numel()
                    optimizer.step()
                    '''End Choreo'''

        #                 optimizer.zero_grad()
        #                 total_loss.backward()
        #                 optimizer.step()

        if args.schedule_clip == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            clip_now = args.clip * ep_ratio

        if args.schedule_adam == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            lr_now = args.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in optimizer.param_groups:
                g['lr'] = lr_now

        if i_episode % args.log_num_episode == 0:
            performance = reward_record[-1]['meanepreward'] * (
                1 - early_done / total_done
            )
            if performance >= Best_performance:
                torch.save(
                    network.state_dict(),
                    ENV_NAME.split('-')[0] + config.file_num +
                    '/CheckPoints/EarlyStopPolicy_Suc_{0}hidden_{'
                    '1}threshold_{2}repeat'.format(
                        config.hid_num, config.thres,
                        str(repeat) + '_temp_best'
                    )
                )
                Best_performance = performance

                if early_done / total_done < 0.1 and reward_record[-1][
                        'meanepreward'] > np.mean(final_100_reward):
                    print('Find Better Novel Policy!')
                    # policy_buffer[str(repeat)+'_'+str(i_episode)] = network

                    torch.save(
                        network.state_dict(),
                        ENV_NAME.split('-')[0] + config.file_num +
                        '/CheckPoints/EarlyStopPolicy_Suc_{0}hidden_{'
                        '1}threshold_{2}repeat'.format(
                            config.hid_num, config.thres,
                            str(repeat) + '_' + str(i_episode)
                        )
                    )
                    # break
            print(
                'early stop proportion:', early_done / total_done,
                'Temp Best Performance:', Best_performance
            )
            print('===============================')
    return reward_record, rwds
