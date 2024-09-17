from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node
    from cls_luigi.tools.constants import MLMAXIMIZATIONMETRICS, MLMINIMIZATIONMETRICS

import logging


class Evaluator:
    def __init__(
            self,
            metric: Literal[MLMAXIMIZATIONMETRICS.metrics, MLMINIMIZATIONMETRICS.metrics],
            punishment_value: int | float,
            component_timeout: Optional[int] = None,
            logger: Optional[logging.Logger] = None
    ) -> None:
        self.metric = metric
        self.component_timeout = component_timeout
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
