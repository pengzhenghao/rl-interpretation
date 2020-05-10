# That's enough!
import gym

from toolbox.env.env_maker import get_env_maker, MiniGridWrapper
from toolbox.env.four_way import FourWayGridWorld, draw, register_four_way

register_four_way()


def register_minigrid():
    if "MiniGrid-Empty-16x16-v0" in [s.id for s in gym.envs.registry.all()]:
        return
    try:
        import gym_minigrid.envs
    except ImportError as e:
        print("Failed to import minigrid environment!")
    else:
        assert "MiniGrid-Empty-16x16-v0" in [
            s.id for s in gym.envs.registry.all()
        ]
        print("Successfully imported minigrid environments!")


# register_minigrid()
