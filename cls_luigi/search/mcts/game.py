import logging
import random
from typing import Dict, List
from cls_luigi.search.core.game import OnePlayerGame


class TreeGrammarGame(OnePlayerGame):
    def __init__(
        self,
        grammar: Dict[str, Dict[str, List[str]]],
        *args,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)
        self.grammar = grammar
        self.start = self.grammar["start"]
        self.non_terminals = self.grammar["non_terminals"]
        self.terminals = self.grammar["terminals"]
        self.rules = self.grammar["rules"]

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def get_initial_state(self):
        return self.start

    def get_valid_actions(
        self,
        state: str,
        parent: str | None = None):

        if state in self.non_terminals:
            return list(self.rules[state].keys()).copy()

        elif state in self.terminals:
            return self.rules[parent][state].copy()

    def get_next_state(
        self,
        action: str
    ):
        return action

    def is_terminal(self, parent, state, action):
        if state == self.get_initial_state():
            return False

        if state in self.non_terminals:
            return False

        elif state in self.terminals:
            if self.rules[parent][state]:
                return False

        if not action:
            return True

        return True

    def get_reward(self, path):
        return random.random()


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
