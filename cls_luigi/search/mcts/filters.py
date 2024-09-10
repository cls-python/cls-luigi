from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Set, Tuple, Optional
if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node


import logging
import networkx as nx
from cls_luigi.search.core.filter import ActionFilter


class UniqueActionFilter(ActionFilter):
    """
    Filter out actions that result in paths that include two different concreate implementations of the same abstract
    class. For example, for an abstract class Scaler is specialized by two concrete classes MinMaxScaler
    and RobustScaler, the filter will remove actions that result in paths that include both MinMaxScaler
    and RobustScaler.
    """
    def __init__(
        self,
        hypergraph: nx.DiGraph,
        abs_classes: Set[str],
        logger: Optional[logging.Logger] = None
    ) -> None:
        super().__init__(logger)

        self.hypergraph = hypergraph
        self.abs_classes = abs_classes
        self.abs_classes_mapping = {}
        self._map_abstract_classes()

    def _map_abstract_classes(
        self
    ) -> None:
        for abs_class in self.abs_classes:
            self.abs_classes_mapping[abs_class] = set(self.hypergraph.successors(abs_class))

    def _get_relevant_states(
        self,
        state: Node,
        abs_classes: Set[str]
    ) -> Set[str]:
        relevant = set()
        for abs_cls in abs_classes:
            for k in self.abs_classes_mapping[abs_cls]:
                if k in state.name:
                    relevant.add(k)

        if state.parent is not None:
            relevant.update(self._get_relevant_states(state.parent, abs_classes))
        return relevant

    def return_valid_actions(
        self,
        state: Node,
        possible_actions: List[Tuple[str]]
    ) -> List[Tuple[str]]:

        relevant_abs_classes = set(state.name).intersection(self.abs_classes)
        if len(possible_actions) > 1 and relevant_abs_classes:
            relevant_states = self._get_relevant_states(state, relevant_abs_classes)
            if relevant_states:
                possible_actions = list(filter(lambda action: relevant_states.issubset(action), possible_actions))
        return possible_actions


class ForbiddenActionFilter(ActionFilter):
    """
    Filter out actions that result in paths that include a forbidden action. forbidden actions are sets of concrete
    implementations that should not be included in the same path.
    """
    def __init__(
        self,
        forbidden_actions: List[Set[str]],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(logger)

        self.forbidden_actions = forbidden_actions

    def return_valid_actions(
        self,
        state: Node,
        possible_actions: List[Tuple[str]]
    ) -> List[Tuple[str]]:

        path = self._get_path(state)
        unique_pipeline_components = set()
        for state in path:
            unique_pipeline_components.update(state.name)

        for pa in possible_actions:
            components = unique_pipeline_components.copy()
            components.update(pa)
            for fa in self.forbidden_actions:
                if fa.issubset(components):
                    possible_actions.remove(pa)
                    break

        return possible_actions
