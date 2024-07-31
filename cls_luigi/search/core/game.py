import abc


class OnePlayerGame(abc.ABC):

    def __init__(self, *args, **kwargs):
        pass

    def get_initial_state(self):
        return NotImplementedError

    def get_valid_actions(self, state):
        return NotImplementedError

    def get_next_state(self, action):
        return NotImplementedError

    def is_terminal(self, parent, state, action):
        return NotImplementedError

    def get_reward(self, path):
        return NotImplementedError
