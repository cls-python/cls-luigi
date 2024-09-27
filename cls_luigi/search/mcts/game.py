from __future__ import annotations
from typing import Tuple, List, Optional, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cls_luigi.search.core.filter import ActionFilter
    from cls_luigi.search.mcts.luigi_pipeline_evaluator import LuigiPipelineEvaluator

import itertools
import logging
import random

import luigi
import networkx as nx


from cls_luigi.tools.constants import MAXIMIZE, MINIMIZE


from cls_luigi.search.core.game import OnePlayerGame
from luigi.task import flatten



class HyperGraphGame(OnePlayerGame):
    def __init__(
        self,
        hypergraph: nx.DiGraph,
        sense: Literal[MAXIMIZE, MINIMIZE] = MAXIMIZE,
        evaluator: LuigiPipelineEvaluator = None,
        filters: Optional[ActionFilter] = None,
        logger: Optional[logging.Logger] = None,
        *args,
        **kwargs
    ) -> None:

        super().__init__(sense, logger, *args, **kwargs)
        self.hypergraph = hypergraph
        self.evaluator = evaluator
        self.filters = filters

    def get_initial_state(
        self
    ) -> Tuple[str]:
        start_state = [
            node for node, data in self.hypergraph.nodes(data=True)
            if data.get("start_node") is True
        ]
        assert len(start_state) == 1, "There should be only one start node!"
        return (start_state[0],)

    def get_valid_actions(
        self,
        state: Tuple[str]
    ) -> List[str]:

        successors = []
        possible_actions = []

        for s in state.name:
            temp_successors = list(self.hypergraph.successors(s))
            if temp_successors:
                successors.append(temp_successors)

        if successors:
            if self.is_terminal_term(state):
                possible_actions.append(tuple(flatten(successors)))

            else:
                if len(successors) == 1:
                    for s in successors[0]:
                        possible_actions.append((s,))
                else:
                    possible_actions = list(itertools.product(*successors))

        self.logger.debug(f"Got valid actions for state {state.name}:\n{possible_actions}")

        # if temp_successors and self.validator.is_valid(s, temp_successors):

        # if (not self.is_terminal_term(state)) and (state.parent is not None):
        #     pass
        if self.filters:
            for f in self.filters:
                possible_actions = f.return_valid_actions(state, possible_actions)

        return possible_actions

    def is_terminal_term(
        self,
        state: Tuple[str]
    ) -> bool:
        """Check if the term is terminal. Terminals here refer to elementary symbols in formal grammar, and doesn't
        last or final node/state/stage in MCTS (see is_final_state()).
        """
        for s in state.name:
            if not self.hypergraph.nodes(data=True)[s].get("terminal_node"):
                return False
        return True

    def is_start(
        self,
        state: Tuple[str]
    ) -> bool:

        return self.hypergraph.nodes(data=True)[state].get("start_node")

    def evaluate(
        self,
        path: List[Tuple[str]]
    ) -> float:
        if self.evaluator:
            task_id, status, reward = self.evaluator.evaluate(path)

            if reward == float("inf"):
                if self.sense == MAXIMIZE:
                        reward = -reward

            else:
                if self.sense == MINIMIZE:
                    reward = -reward

            return task_id, status, reward

        self.logger.warning(f"No Evaluator found! Returning random reward for now!\nThis is only for debugging!")
        return None, "NOEVALUATOR", random.random()

    def is_final_state(
        self,
        state: Tuple[str]
    ) -> bool:
        return self.get_valid_actions(state) == []

    def get_luigi_pipeline(
        self,
        path: List[Tuple[str]]
    ) -> luigi.Task:
        return self.evaluator._get_luigi_task(path)

    def reset(self):
        self.evaluator.reset()


if __name__ == "__main__":
    tree_grammar = {
        "start": "CLF",
        "non_terminals": ["CLF", "FP", "Scaler", "Imputer", "Data"],
        "terminals": ["rf", "knn", "pca", "polynomial_feats", "standard", "minmax", "mean", "median", "csv"],
        "rules": {
            "CLF": {"rf": ["FP"], "knn": ["FP"]},
            "FP": {"pca": ["Scaler"], "polynomial_feats": ["Scaler"]},
            "Scaler": {"standard": ["Imputer"], "minmax": ["Imputer"]},
            "Imputer": {"mean": ["Data"], "median": ["Data"]},
            "Data": {"csv": []}
        }
    }
