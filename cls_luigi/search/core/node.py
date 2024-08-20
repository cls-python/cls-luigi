import abc
import logging
from typing import Type, Tuple


class NodeBase(abc.ABC):
    def __init__(
        self,
        name: str | Tuple[str],
        logger: logging.Logger = None,
        **kwargs,

    ) -> None:
        self.expandable_actions = None
        self.parent: str
        self.name: str
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.children = []
        self.visits = 0
        self.reward = 0
        self.name = name

    def select(
        self
    ) -> Type['NodeBase']:
        ...

    def expand(
        self
    ) -> Type['NodeBase']:
        ...

    def simulate(
        self,
        path
    ) -> float:
        ...

    def backpropagate(
        self,
        reward: float
    ) -> None:
        ...

    def is_fully_expanded(
        self
    ) -> bool:
        ...
