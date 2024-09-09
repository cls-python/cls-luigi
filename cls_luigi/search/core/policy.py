from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cls_luigi.search.mcts.game import OnePlayerGame
    from cls_luigi.search.mcts.node import Node

import abc
import logging


class SelectionPolicy(abc.ABC):
    def __init__(
        self,
        node: Node,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> None:

        self.node = node
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def select(
        self
    ) -> Node:
        ...

    def get_score(
        self,
        node: Node
    ) -> float:
        ...

    def _regularization(self):
        return None


class ExpansionPolicy(abc.ABC):
    def __init__(
        self,
        node: Node,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> None:
        self.node = node
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def get_action(
        self
    ) -> Node:
        ...


class SimulationPolicy(abc.ABC):
    def __init__(
        self,
        game: OnePlayerGame,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> None:
        self.game = game
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def get_action(
        self,
        state: str
    ) -> str:
        ...
