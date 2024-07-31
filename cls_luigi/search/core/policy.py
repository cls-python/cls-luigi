import abc
import logging
from typing import Type

from cls_luigi.search.core.game import OnePlayerGame
from cls_luigi.search.core.node import NodeBase


class SelectionPolicy(abc.ABC):
    def __init__(
        self,
        node: NodeBase,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:

        self.node = node
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def select(
        self
    ) -> Type[NodeBase]:
        ...

    def get_score(
        self,
        node: Type[NodeBase]
    ) -> float:
        ...


class ExpansionPolicy(abc.ABC):
    def __init__(
        self,
        node: NodeBase,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:
        self.node = node
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def get_action(
        self
    ) -> Type[NodeBase]:
        ...


class SimulationPolicy(abc.ABC):
    def __init__(
        self,
        game: OnePlayerGame,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:
        self.game = game
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def get_action(
        self,
        state: NodeBase
    ) -> Type[NodeBase]:
        ...
