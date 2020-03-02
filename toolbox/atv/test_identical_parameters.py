import shutil
import tempfile
import unittest

import numpy as np
from ray import tune

from toolbox import initialize_ray
from toolbox.atv.identical_parameters import get_dynamic_trainer


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
    trainer = get_dynamic_trainer(algo, 10000, "BipedalWalker-v2")(
        config={"env": "BipedalWalker-v2"})

    if algo in ["ES", "ARS"]:
        tw = {k: v for k, v in trainer.policy.variables.get_weights().items()
              if "value" not in k}
    elif algo in ["PPO", "A2C", "A3C", "IMPALA"]:
        tw = {k: v for k, v in trainer.get_weights()['default_policy'].items()
              if "value" not in k}
    # elif algo == "TD3":
    #     tw = {
    #         k.split("policy/")[-1].replace("dense", "fc"): v
    #         for k, v in trainer.get_weights()['default_policy'].items()
    #         if "/policy/" in k
    #     }

    rw = {
        k: v for k, v in
        trainer._reference_agent_weights['default_policy'].items()
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
    config = {"env": "BipedalWalker-v2"}
    if algo == "ES":
        config['num_workers'] = 2
    dir_path = tempfile.mkdtemp()
    trainer = get_dynamic_trainer(algo, 10000, "BipedalWalker-v2")
    ret = tune.run(
        trainer,
        local_dir=dir_path,
        stop={"training_iteration": 10},
        config=config,
        verbose=2,
        max_failures=0
    )
    shutil.rmtree(dir_path, ignore_errors=True)
    return ret


def test_reference_consistency():
    initialize_ray(test_mode=True, local_mode=False)
    algos = ["PPO", "ES", "A2C", "A3C", "IMPALA", "ARS"]
    rws = {}
    for i, algo in enumerate(algos):
        trainer = get_dynamic_trainer(algo, 10000, "BipedalWalker-v2")(config={
            "env": "BipedalWalker-v2",
            "seed": i * 1000 + 789
        })
        rw = {
            k: v for k, v in
            trainer._reference_agent_weights['default_policy'].items()
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

    # def test_td3(self):
    #     _test_basic("TD3")

    def test_a2c(self):
        _test_basic("A2C")

    def test_a3c(self):
        _test_basic("A3C")

    def test_impala(self):
        _test_basic("IMPALA")

    def test_ars(self):
        _test_basic("ARS")


class BlackBoxTest(unittest.TestCase):
    def test_ppo(self):
        _test_blackbox("PPO")

    def test_es(self):
        _test_blackbox("ES")

    # def test_td3(self):
    #     _test_blackbox("TD3")

    def test_a2c(self):
        _test_blackbox("A2C")

    def test_a3c(self):
        _test_blackbox("A3C")

    def test_impala(self):
        _test_blackbox("IMPALA")

    def test_ars(self):
        _test_blackbox("ARS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
