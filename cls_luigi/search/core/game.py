import abc
import logging
from typing import Optional


class OnePlayerGame(abc.ABC):
    minimization_problem = None

    def __init__(
        self,
        minimization_problem: bool,
        logger: Optional[logging.Logger] = None,
        *args,
        **kwargs
    ) -> None:
        self.minimization_problem = minimization_problem
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def get_initial_state(self):
        ...

    def get_valid_actions(self, state):
        ...

    def is_terminal_term(self, state):
        ...

    def get_reward(self, path):
        ...

    def is_final_state(self, state):
        ...

    def is_start(self, Optional):
        pass
