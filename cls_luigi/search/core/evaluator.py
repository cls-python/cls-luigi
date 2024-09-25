from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Literal, Union

if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node
    from cls_luigi.tools.constants import MLMAXIMIZATIONMETRICS, MLMINIMIZATIONMETRICS

import logging


class Evaluator:
    def __init__(
            self,
            metric: Literal[MLMAXIMIZATIONMETRICS, MLMINIMIZATIONMETRICS],
            punishment_value: int | float,
            pipeline_timeout: Optional[int] = None,
            task_timeout: Optional[Union[int, float]] = None,
            logger: Optional[logging.Logger] = None
    ) -> None:
        self.metric = metric
        self.component_timeout = task_timeout
        self.pipeline_timeout = pipeline_timeout if pipeline_timeout is not None else float("inf")
        self.punishment_value = punishment_value
        self.evaluated = {}
        self.failed = {}
        self.timed_out = {}
        self.missing_score = {}
        self.not_found = []

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(
            self,
            path: List[Node]
    ) -> float | int:
        raise NotImplementedError(f"Method {self.__class__.__name__}.evaluate() not implemented")
