"""Copy from https://github.com/ray-project/ray/blob/master/python/ray/tests
/test_unreconstructable_errors.py"""
import numpy as np
import unittest

import ray

MB = 1024 * 1024

class TestUnreconstructableErrors(unittest.TestCase):
    def setUp(self):
        ray.init(
            num_cpus=1,
            object_store_memory=150 * MB,
            redis_max_memory=10000000)

    def tearDown(self):
        ray.shutdown()

    def testDriverPutEvictedCannotReconstruct(self):
        x_id = ray.put(np.zeros(1 * MB), weakref=True)
        ray.get(x_id)
        for _ in range(20):
            ray.put(np.zeros(10 * MB))
        self.assertRaises(ray.exceptions.UnreconstructableError,
                          lambda: ray.get(x_id))

    def testLineageEvictedReconstructionFails(self):
        @ray.remote
        def f(data):
            return 0

        x_id = f.remote(None)
        ray.get(x_id)
        for _ in range(400):
            ray.get([f.remote(np.zeros(10000)) for _ in range(50)])
        self.assertRaises(ray.exceptions.UnreconstructableError,
                          lambda: ray.get(x_id))


if __name__ == "__main__":
    # unittest.main(verbosity=2)
    ray.init(object_store_memory=int(250e6)) # 250 Mb


    @ray.remote
    def identity(vectors):
        return [ray.put(ray.get(vec)) for vec in vectors]


    vectors = [ray.put(np.empty(int(0.5 * 1e6), dtype=np.uint8)) for _ in range(200)]
    i = 0
    while True:
        vectors = ray.get(identity.remote(vectors))
        i += 1
        print(i)
