import abc
import logging


class OnePlayerGame(abc.ABC):

    minimization_problem = None

    def __init__(self, logger, *args, **kwargs):
        if logger is None:
            self.logger = logging.getLogger(__name__)

    def get_initial_state(self):
        return NotImplementedError

    def get_valid_actions(self, state):
        return NotImplementedError

    def is_terminal_term(self, state):
        return NotImplementedError

    def get_reward(self, path):
        return NotImplementedError

    def is_final_state(self, state):
        pass
