from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node

import abc
import logging


class TreeBase(abc.ABC):
    def __init__(
        self,
        root: Node,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> None:

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.root = root

    def draw_tree(self, *args, **kwargs):
        ...

    def save(self, *args, **kwargs):
        ...
