# copied from https://pastebin.com/pfnd5haf
from gym.core import Wrapper
from pickle import dumps, loads
from collections import namedtuple

# a container for get_result function below. Works just like tuple,
# but prettier
ActionResult = namedtuple("action_result", (
"snapshot", "observation", "reward", "is_done", "info"))


class WithSnapshots(Wrapper):
    """
   Creates a wrapper that supports saving and loading environemnt states.
   Required for planning algorithms.

   This class will have access to the core environment as self.env, e.g.:
   - self.env.reset()           #reset original env
   - self.env.ale.cloneState()  #make snapshot for atari. load with .restoreState()
   - ...

   You can also use reset, step and render directly for convenience.
   - s, r, done, _ = self.step(action)   #step, same as self.env.step(action)
   - self.render(close=True)             #close window, same as self.env.render(close=True)
   """

    def get_snapshot(self):
        """
       :returns: environment state that can be loaded with load_snapshot
       Snapshots guarantee same env behaviour each time they are loaded.

       Warning! Snapshots can be arbitrary things (strings, integers, json, tuples)
       Don't count on them being pickle strings when implementing MCTS.

       Developer Note: Make sure the object you return will not be affected by
       anything that happens to the environment after it's saved.
       You shouldn't, for example, return self.env.
       In case of doubt, use pickle.dumps or deepcopy.

       """
        # self.render(
        #     close=True)  # close popup windows since we can't pickle them
        return dumps(self.env)

    def load_snapshot(self, snapshot):
        """
       Loads snapshot as current env state.
       Should not change snapshot inplace (in case of doubt, deepcopy).
       """

        assert not hasattr(self, "_monitor") or hasattr(self.env,
                                                        "_monitor"), "can't backtrack while recording"

        # self.render(
        #     close=True)  # close popup windows since we can't load into them
        self.env = loads(snapshot)

    def get_result(self, snapshot, action):
        """
       A convenience function that
       - loads snapshot,
       - commits action via self.step,
       - and takes snapshot again :)

       :returns: next snapshot, next_observation, reward, is_done, info

       Basically it returns next snapshot and everything that env.step would have returned.
       """
        self.load_snapshot(snapshot)

        obs, r, done, info = self.step(action)

        new_snapshot = self.get_snapshot()

        return ActionResult(new_snapshot, obs, r, done, info)

if __name__ == '__main__':
    pass