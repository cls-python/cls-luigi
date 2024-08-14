import logging
import random
from typing import Dict, List

import networkx as nx

from cls_luigi.search.core.game import OnePlayerGame


class HyperGraphGame(OnePlayerGame):
    def __init__(self, g: nx.DiGraph, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.G = g

    def get_initial_state(self):
        start_state = [
            node for node, data in self.G.nodes(data=True)
            if data.get("start_node") is True
        ]
        assert len(start_state) == 1, "There should be only one start node!"
        return start_state[0]

    def get_valid_actions(self, state):
        return list(self.G.successors(state))

    def is_terminal(self, state):
        return self.G.nodes(data=True)[state].get("terminal_node")

    def is_start(self, state):
        return self.G.nodes(data=True)[state].get("start_node")

    def get_reward(self, path):
        return random.random()

    def is_final_state(self, state):
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
