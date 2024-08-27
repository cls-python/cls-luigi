import itertools
import random
from typing import Tuple, List

import luigi
import networkx as nx

from cls_luigi.search.core.game import OnePlayerGame
from cls_luigi.search.mcts.evaluator import Evaluator
from luigi.task import flatten


class HyperGraphGame(OnePlayerGame):
    def __init__(
        self,
        g: nx.DiGraph,
        evaluator: Evaluator | None = None,
        logger=None,
        *args,
        **kwargs
    ) -> None:

        super().__init__(logger, *args, **kwargs)
        self.G = g
        self.evaluator = evaluator

    def get_initial_state(
        self
    ) -> Tuple[str]:
        start_state = [
            node for node, data in self.G.nodes(data=True)
            if data.get("start_node") is True
        ]
        assert len(start_state) == 1, "There should be only one start node!"
        return (start_state[0],)

    def get_valid_actions(
        self,
        state: Tuple[str]
    ) -> List[str]:

        successors = []
        valid_actions = []

        for s in state:
            _successors = list(self.G.successors(s))
            if _successors:
                successors.append(_successors)

        if successors:
            if self.is_terminal_term(state):
                valid_actions.append(tuple(flatten(successors)))

            else:
                if len(successors) == 1:
                    for s in successors[0]:
                        valid_actions.append((s,))
                else:
                    valid_actions = list(itertools.product(*successors))

        return valid_actions

    def is_terminal_term(
        self,
        state: Tuple[str]
    ) -> bool:
        if isinstance(state, tuple):
            state = state[0]
        return self.G.nodes(data=True)[state].get("terminal_node")

    def is_start(
        self,
        state: Tuple[str]
    ) -> bool:

        return self.G.nodes(data=True)[state].get("start_node")

    def get_reward(
        self,
        path: List[Tuple[str]]
    ) -> float:
        if self.evaluator:
            return self.evaluator.evaluate(path)
        return random.random()

    def is_final_state(
        self,
        state: Tuple[str]
    ) -> bool:
        return self.get_valid_actions(state) == []

    def get_luigi_pipeline(
        self,
        path: List[Tuple[str]]
    ) -> luigi.Task:
        return self.evaluator.get_luigi_pipeline(path)


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
