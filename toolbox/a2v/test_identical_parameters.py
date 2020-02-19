import shutil
import tempfile
import unittest

import numpy as np
from ray import tune

from toolbox import initialize_ray
from toolbox.a2v.identical_parameters import get_dynamic_trainer


def assert_equal(arr1, arr2, k=""):
    comp1 = arr1
    comp2 = arr2
    if arr1.shape != arr2.shape:
        axis = arr1.ndim - 1
        if arr1.shape[axis] > arr2.shape[axis]:
            comp1 = np.split(arr1, 2, axis=axis)[0]
        else:
            comp2 = np.split(arr2, 2, axis=axis)[0]
    assert comp1.shape == comp2.shape, (comp1.shape, comp2.shape, k)
    np.testing.assert_equal(comp1, comp2)


def _test_basic(algo):
    initialize_ray(test_mode=True)
    trainer = get_dynamic_trainer(algo)(config={
        "env": "BipedalWalker-v2",
        "init_seed": 10000,
    })

    if algo == "ES":
        tw = {k: v for k, v in
              trainer.policy.variables.get_weights().items()}
    elif algo == "PPO":
        tw = {k: v for k, v in trainer.get_weights()['default_policy'].items()
              if "value" not in k}
    elif algo == "TD3":
        tw = {
            k.split("policy/")[-1].replace("dense", "fc"): v
            for k, v in trainer.get_weights()['default_policy'].items()
            if "/policy/" in k
        }

    rw = {
        k: v for k, v in
        trainer._reference_agent.get_weights()['default_policy'].items()
        if "value" not in k}

    assert len(tw) == len(rw)

    twk = list(tw.keys())
    rwk = list(rw.keys())

    for i in range(len(tw)):
        arr1 = tw[twk[i]]
        arr2 = rw[rwk[i]]

        assert_equal(arr1, arr2)


def _test_blackbox(algo):
    initialize_ray(test_mode=True)
    config = {"env": "BipedalWalker-v2", "init_seed": 10000}
    if algo == "ES":
        config['num_workers'] = 2
    dir_path = tempfile.mkdtemp()
    trainer = get_dynamic_trainer(algo)
    ret = tune.run(
        trainer,
        local_dir=dir_path,
        stop={"timesteps_total": 1000},
        config=config,
        verbose=2,
        max_failures=0
    )
    shutil.rmtree(dir_path, ignore_errors=True)
    return ret


def test_reference_consistency():
    initialize_ray(test_mode=True)
    algos = ["PPO", "ES", "TD3"]
    rws = {}
    for i, algo in enumerate(algos):
        trainer = get_dynamic_trainer(algo)(config={
            "env": "BipedalWalker-v2",
            "init_seed": 10000,
            "seed": i * 1000 + 789
        })
        rw = {
            k: v for k, v in
            trainer._reference_agent.get_weights()['default_policy'].items()
            if "value" not in k
        }
        rws[algo] = rw
    ks = list(rws)
    first_weight_dict = next(iter(rws.values()))
    for weight_name in first_weight_dict.keys():
        print("Current weight name: ", weight_name)
        for weight_dict_name in ks[1:]:
            weight_dict = rws[weight_dict_name]
            assert_equal(first_weight_dict[weight_name],
                         weight_dict[weight_name])


class BasicTest(unittest.TestCase):
    def test_ppo(self):
        _test_basic("PPO")

    def test_es(self):
        _test_basic("ES")

    def test_td3(self):
        _test_basic("TD3")


class BlackBoxTest(unittest.TestCase):
    def test_ppo(self):
        _test_blackbox("PPO")

    def test_es(self):
        _test_blackbox("ES")

    def test_td3(self):
        _test_blackbox("TD3")


if __name__ == "__main__":
    unittest.main(verbosity=2)