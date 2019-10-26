import time

import numpy as np
import ray

from toolbox.abstract_worker import WorkerManagerBase, WorkerBase
from toolbox.utils import initialize_ray

MB = 1024 * 1024


def test_heavy_memory_usage():
    # initialize_ray(test_mode=True, object_store_memory=4000 * MB)
    initialize_ray(test_mode=True)

    num = 50
    delay = 0
    num_workers = 16

    class TestWorker(WorkerBase):
        def __init__(self):
            self.count = 0

        def run(self):
            time.sleep(delay)
            self.count += 1
            print(self.count, ray.cluster_resources())
            arr = np.empty((10 * MB), dtype=np.uint8)
            return self.count, arr

    class TestManager(WorkerManagerBase):
        def __init__(self):
            super(TestManager, self).__init__(
                num_workers, TestWorker, num, 1, 'test')

        def count(self, index):
            self.submit(index)

    tm = TestManager()
    for i in range(num):
        tm.count(i)

    ret = tm.get_result()
    return ret


if __name__ == '__main__':
    ret = test_heavy_memory_usage()
