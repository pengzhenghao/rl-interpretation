from collections import OrderedDict

import numpy as np


def _dist(shape, sample_size, random_state=None):
    assert isinstance(shape, list)
    assert shape[0] is None
    new_shape = [sample_size] + shape[1:]
    pack = random_state if random_state else np.random
    return pack, new_shape


def one_dist(shape, sample_size, random_state=None):
    pack, new_shape = _dist(shape, sample_size, random_state)
    return np.ones(new_shape, dtype=np.float32)


def binary_dist(shape, sample_size, random_state=None):
    pack, new_shape = _dist(shape, sample_size, random_state)
    return pack.choice([0, 1], new_shape).astype(np.float32)


def normal_dist(shape, sample_size, random_state=None):
    pack, new_shape = _dist(shape, sample_size, random_state)
    return pack.normal(loc=1.0, scale=1.0, size=new_shape)


def dual_replay(
        agent,
        env_state_list,
        observations,
        env_maker,
        probe_mask_name,
        sample_size=100,
        seed=0,
        distribution="binary"
):
    # FIXME The seed setting is useless here.
    # when you run this funcion mulitple times, you always get different
    # return.
    # I am not sure what's going on.

    assert agent._name == "PPOWithMask"

    mask_info = agent.get_mask_info()

    rs = np.random.RandomState(seed)

    buffer_size = len(env_state_list)
    indices = rs.choice(
        buffer_size, sample_size
    )  # uniformaly choose indices from dataset

    env = env_maker()
    env.reset()

    if distribution == "normal":
        dist_callback = normal_dist
    elif distribution == "binary":
        dist_callback = binary_dist
    else:
        raise ValueError("You give a wrong distribution name.")

    # make the mask_dict
    # firstly, we only want to see the impact of second layer.
    mask_dict = {
        name: one_dist(shape, sample_size, rs)
        for name, shape in mask_info.items()
    }
    assert isinstance(mask_info, OrderedDict)

    probe_layer_shape = mask_info[probe_mask_name]
    mask_dict[probe_mask_name] = dist_callback(
        probe_layer_shape, sample_size, rs
    )

    obs = observations[indices]
    act, _, infos = agent.get_policy().compute_actions(
        obs, mask_batch=mask_dict
    )

    new_ob_list = []
    rew_list = []
    done_list = []

    count = 1
    for act_id, ind in enumerate(indices):
        if count % 500 == 0:
            print("Current Steps: [{}/{}]".format(count, sample_size))
        count += 1
        obs = observations[ind]

        env_state = env_state_list[ind]
        env.set_state_wrap(env_state)

        new_ob, reward, done, _ = env.step(act[act_id])
        new_ob_list.append(new_ob)
        rew_list.append(reward)
        done_list.append(done)

    ret = {
        "obs": observations[indices],
        "new_action": np.stack(act),
        "new_next_obs": np.stack(new_ob_list),
        "new_reward": np.array(rew_list),
        "new_done": np.array(done_list),
        "indices": indices,
        "mask": mask_dict
    }

    return ret
