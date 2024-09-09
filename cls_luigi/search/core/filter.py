from __future__ import annotations

from typing import Tuple, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node
import abc
import logging

class ActionFilter(abc.ABC):
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,

    ):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def return_valid_actions(
        self,
        state,
        possible_actions: List[Tuple[str]]
    ) -> List[Tuple[str]]:
        ...

    def _get_path(
        self,
        state: List[Node]
    ):
        path = [state]
        if state.parent is not None:
            path.extend(self._get_path(state.parent))
        return list(reversed(path))
