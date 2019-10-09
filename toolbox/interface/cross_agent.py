"""
This file contain the interface to cross-agent analysis.
"""


class CrossAgentAnalyst:
    """
    The logic of this interface:

    1. feed data
    2. try the methods whatever you like

    """

    methods = {
        "represent": ["fft_represent", "naive_represent"],
        "similarity": ["cka_similarity"],
        "distance": ["js_distance", "cka_distance", "naive_represent_distance"]
    }

    def __init__(self):

        self.computed_result = {}
        for k, name_list in self.methods.items():
            self.computed_result[k] = {
                method_name: None for method_name in name_list
            }

        self.rollout_dataset = None

    def _check_input(self):
        if self.rollout_dataset is None:
            print("Data is not loaded! Please call feed(...) before "
                  "doing anything!")
            return False
        return True

    def feed(self):
        """
        1. store the data
        2. build the joint dataset
        3. lazy's replay the necessary rollout

        :return:
        """

    def naive_represent(self):
        pass

    def fft_represent(self):
        pass

    def cka_similarity(self):
        pass


