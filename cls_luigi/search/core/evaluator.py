from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node

import logging

class Evaluator:
    def __init__(
        self,
        punishment_value: int | float,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.punishment_value = punishment_value
        self.evaluated = []
        self.failed = {}

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(
        self,
        path: List[Node]
    ) -> float | int:
        raise NotImplementedError("Method evaluate not implemented")
