import logging
from typing import Dict, List, Type, Any

from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.core.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy
from cls_luigi.search.core.tree import TreeBase
from cls_luigi.search.mcts.game import TreeGrammarGame, OnePlayerGame
from cls_luigi.search.mcts.tree import MCTSTreeWithGrammar
from cls_luigi.search.mcts.node import Node
from cls_luigi.search.mcts.policy import UCB1, RandomExpansion, RandomSimulation


class SP_MCTS:
    """Pure implementation of a Single player Monte Carlo Tree Search ."""

    def __init__(
        self,
        grammar: Dict[str, Dict[str, List[str]] | str | List[str]],
        parameters: Dict[str, Any],
        game_class: Type[OnePlayerGame] = TreeGrammarGame,
        selection_policy: Type[SelectionPolicy] = UCB1,
        expansion_policy: Type[ExpansionPolicy] = RandomExpansion,
        simulation_policy: Type[SimulationPolicy] = RandomSimulation,
        tree_cls: Type[TreeBase] = MCTSTreeWithGrammar,
        logger: logging.Logger = None
    ) -> None:

        self.grammar = grammar
        self.game = game_class(self.grammar)
        self.parameters = parameters
        self.selection_policy = selection_policy
        self.expansion_policy = expansion_policy
        self.simulation_policy = simulation_policy
        self.tree = tree_cls(root=self.init_root(), grammar=grammar)
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def init_root(
        self
    ) -> NodeBase:
        root_node = Node(
            game=self.game,
            params=self.parameters,
            name=self.game.get_initial_state(),
            selection_policy_cls=self.selection_policy,
            expansion_policy_cls=self.expansion_policy,
            simulation_policy_cls=self.simulation_policy)
        logging.debug(f"Initialized root node {root_node.name}")

        return root_node

    def run(
        self
    ) -> None:
        self.logger.debug("Running SP-MCTS for {} iterations".format(self.parameters["num_iterations"]))

        paths = []

        for _ in range(self.parameters["num_iterations"]):
            path = []
            logging.debug("======== Iteration: {}".format(_))
            node = self.tree.get_root()
            path.append(node)

            while node.is_fully_expanded():
                self.logger.debug(f"========= fully expanded: {node.name}")
                node = node.select()
                path.append(node)
                self.logger.debug(f"========= selected new node: {node.name}")

            is_terminal = False
            if node.parent:
                is_terminal = self.game.is_terminal(node.parent.name, node.name, node.action_taken)

            if not is_terminal:
                self.logger.debug(f"========= not terminal: {node.name}")
                node = node.expand()
                self.tree.add_node(node)
                self.logger.debug(f"========= expanded:{node.name}")
                reward = node.simulate(path)
            else:
                self.logger.debug(f"========= terminal:{node.name}")
                if path not in paths:
                    paths.append(path)
                reward = self.game.get_reward(path)

            node.backprop(reward)

        # print(paths)
        # print("=================================================\n")
        self.tree.draw_tree()
        # best = self.get_best_path()
        # print("Best path:", best)
        # print("=================================================\n")

    def get_best_path(
        self
    ) -> List[Node]:

        node = self.tree.get_root()
        path = [node]
        while len(node.children) > 0:
            node = node.select()
            path.append(node)
        return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tree_grammar = {
        "start": "CLF",
        "non_terminals": ["CLF", "FP", "Scaler", "Imputer", "Data"],
        "terminals": ["rf", "knn", "pca", "poly", "standard", "minmax", "mean", "median", "csv"],
        "rules": {
            "CLF": {"rf": ["FP"], "knn": ["FP"]},
            "FP": {"pca": ["Scaler"], "poly": ["Scaler"]},
            "Scaler": {"standard": ["Imputer"], "minmax": ["Imputer"]},
            "Imputer": {"mean": ["Data"], "median": ["Data"]},
            "Data": {"csv": []}
        }
    }

    params = {
        "num_iterations": 100,
        "exploration_param": 0.5
    }

    mcts = SP_MCTS(
        game_class=TreeGrammarGame,
        grammar=tree_grammar,
        parameters=params,
        selection_policy=UCB1,
    )
    mcts.run()
