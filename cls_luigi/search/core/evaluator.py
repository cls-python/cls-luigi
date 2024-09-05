import logging
from typing import List


class Evaluator:
    def __init__(
        self,
        punishment_value: int | float,
        logger: logging.Logger = None
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
        path: List["NodeBase"]
    ) -> float | int:
        raise NotImplementedError("Method evaluate not implemented")
