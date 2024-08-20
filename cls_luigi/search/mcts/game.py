import itertools
import random
from typing import Tuple, List

import networkx as nx

from cls_luigi.search.core.game import OnePlayerGame


class HyperGraphGame(OnePlayerGame):
    def __init__(
        self,
        g: nx.DiGraph,
        logger=None,
        *args,
        **kwargs
    ) -> None:

        super().__init__(logger, *args, **kwargs)
        self.G = g

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

        valid_actions = []

        if len(state) == 1:
            if not self.is_terminal_term(state):
                valid_actions.extend(list(map(lambda x: (x,), list(self.G.successors(state[0])))))
                # return valid_actions

            else:
                successors = tuple(self.G.successors(state[0]))
                if successors:
                    valid_actions.append(successors)

        else:
            for s in state:
                successors = list(self.G.successors(s))
                if successors:
                    valid_actions.append(successors)
            if len(valid_actions) > 1:
                valid_actions = list(itertools.product(*valid_actions))
            elif len(valid_actions) == 1:
                valid_actions = [(valid_actions[0][0],)]

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

        return random.random()

    def is_final_state(
        self,
        state: Tuple[str]
    ) -> bool:

        return self.get_valid_actions(state) == []


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
