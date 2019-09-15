import copy

import numpy as np
import ray

from toolbox.ablate.process_diss import ablate_multiple_units, AblationWorker, ABLATE_LAYER_NAME, get_agent_layer_names
from toolbox.utils import build_env, initialize_ray
from toolbox.test.utils import get_ppo_agent


def test_ablation_worker_replay():
    from toolbox.test.utils import get_test_agent_config, load_test_agent_rollout
    from toolbox.utils import initialize_ray

    initialize_ray(test_mode=True)
    worker = AblationWorker.remote()
    obs_list = load_test_agent_rollout()
    obs = np.concatenate([o['obs'] for o in obs_list])

    config = get_test_agent_config()
    worker.reset.remote(
        run_name=config['run_name'],
        ckpt=config['ckpt'],
        env_name=config['env_name'],
        env_maker=build_env,
        agent_name="Test Agent",
        worker_name="Test Worker"
    )

    result = copy.deepcopy(ray.get(worker.replay.remote(obs=obs)))

    result2 = copy.deepcopy(
        ray.get(
            worker.replay.remote(
                obs=obs, layer_name=ABLATE_LAYER_NAME, unit_index=10
            )
        )
    )

    result3 = copy.deepcopy(
        ray.get(
            worker.replay.remote(
                obs=obs,
                layer_name="default_policy/default_model/fc_out",
                unit_index=20
            )
        )
    )

    return result, result2, result3


def test_ablate_multiple_units():
    initialize_ray(test_mode=True)
    agent = get_ppo_agent()
    print(get_agent_layer_names(agent))
    mapping = {
        "default_policy/fc_out": [10, 11, 12, 0],
        "default_policy/fc_2": [12, 54, 102, 0]
    }
    ablated_agent = ablate_multiple_units(agent, mapping)
    weights = ablated_agent.get_policy()._variables.get_weights()
    print("Current layer name: ", weights.keys())
    for layer, units in mapping.items():
        layer += "/kernel"
        vector = weights[layer].mean(1)
        assert np.all(vector[units] == 0)
    print("Test passed!")
    return ablated_agent


if __name__ == '__main__':
    test_ablate_multiple_units()
