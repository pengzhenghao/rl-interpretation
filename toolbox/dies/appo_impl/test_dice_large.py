from toolbox.dies.appo_impl.test_dice_appo import _test_dice

if __name__ == '__main__':
    _test_dice({
        "train_batch_size": 4000,
        "sample_batch_size": 200,
        "num_workers": 5,
        "num_agents": 5
    }, t=10000)
