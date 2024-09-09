from __future__ import annotations
import abc
import logging
from typing import Tuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node


class NodeBase(abc.ABC):
    def __init__(
        self,
        name: str | Tuple[str],
        logger: Optional[logging.Logger] = None,
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
        self.sum_rewards = 0
        self.name = name
        self.explanations = []

    def select(
        self
    ) -> Node:
        ...

    def expand(
        self
    ) -> Node:
        ...

    def simulate(
        self,
        path
    ) -> float:
        ...

    def backprop(
        self,
        reward: float
    ) -> None:
        ...

    def is_fully_expanded(
        self
    ) -> bool:
        ...
