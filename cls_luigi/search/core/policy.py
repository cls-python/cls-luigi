import abc
from typing import Type

from cls_luigi.search.core.game import OnePlayerGame
from cls_luigi.search.core.node import NodeBase


class SelectionPolicy(abc.ABC):
    def __init__(
        self,
        node: NodeBase,
        **kwargs
    ) -> None:
        self.node = node

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
        **kwargs
    ) -> None:
        self.node = node

    def get_action(
        self
    ) -> Type[NodeBase]:
        ...


class SimulationPolicy(abc.ABC):
    def __init__(
        self,
        game: OnePlayerGame,
        **kwargs
    ) -> None:
        self.game = game

    def get_action(
        self,
        state: NodeBase
    ) -> Type[NodeBase]:
        ...
