from toolbox.dece.dece import DECETrainer
from toolbox.marl.test_extra_loss import _base


def test_dece(local_mode=False):
    _base(
        trainer=DECETrainer,
        local_mode=local_mode,
        extra_config={},
        env_name="Pendulum-v0",
        t=1000
    )


if __name__ == '__main__':
    test_dece(local_mode=False)
