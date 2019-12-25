"""Implement a diversity-seeking + evolution-strategy algorithm.

We need to implement the following things:
1. We need to update, swap or mutate the weight of all existing policies:
    (1). Copy the weight of the best policy to all other.
2. TNB loss as the update.

A serious problem is that the novelty-value-network is obviously hard to
train: each time the policies are mutually swapped, the novelty value network
(novelty reward) is changed drastically. This required the
novelty-value-network to fine-tune.

What's more, the novelty value network make more trainable parameters, and when
it is not fully functional, the signal sent to the policy network is chaos.

How to solve it?

1. Share the value network with the NVN and only use an extra layer to serve
as the NVN output layer. However, the messed signal of novelty reward may also
harm the performance of the original value network.

2. Remove the whole actor-critic framework for diversity-seeking, but instead
use the pure loss as diversity-seeking supervision.

"""