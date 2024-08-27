import logging
import pickle
from typing import Dict, List, Type, Any

import luigi
from luigi import Task

from cls_luigi.grammar.hypergraph import get_hypergraph_dict_from_tree_grammar, plot_hypergraph_components, \
    build_hypergraph
from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.core.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy
from cls_luigi.search.core.tree import TreeBase
from cls_luigi.search.helpers import set_seed
from cls_luigi.search.mcts.game import HyperGraphGame, OnePlayerGame
from cls_luigi.search.mcts.tree import MCTSTreeWithGrammar
from cls_luigi.search.mcts.node import Node, NodeFactory
from cls_luigi.search.mcts.policy import UCT, RandomExpansion, RandomSimulation


class PureSinglePlayerMCTS:
    """Pure implementation of a Single player Monte Carlo Tree Search ."""

    def __init__(
        self,
        parameters: Dict[str, Any],
        game: Type[OnePlayerGame],
        selection_policy: Type[SelectionPolicy] = UCT,
        expansion_policy: Type[ExpansionPolicy] = RandomExpansion,
        simulation_policy: Type[SimulationPolicy] = RandomSimulation,
        tree_cls: Type[TreeBase] = MCTSTreeWithGrammar,
        node_factory_cls: NodeFactory = NodeFactory,
        logger: logging.Logger = None,
    ) -> None:

        self.game = game
        self.node_factory_cls = node_factory_cls
        self.node_factory = self.node_factory_cls(self.game)
        self.parameters = parameters
        self.selection_policy = selection_policy
        self.expansion_policy = expansion_policy
        self.simulation_policy = simulation_policy
        self.tree = tree_cls(root=self.get_root_node(), hypergraph=self.game.G)
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def get_root_node(
        self
    ) -> Type[NodeBase]:

        return self.node_factory.create_node(
            params=self.parameters,
            name=self.game.get_initial_state(),
            selection_policy_cls=self.selection_policy,
            expansion_policy_cls=self.expansion_policy,
            simulation_policy_cls=self.simulation_policy,
            node_factory=self.node_factory,
        )

    def run(
        self
    ) -> luigi.Task:
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

            is_final_state = self.game.is_final_state(node.name)

            if not is_final_state:
                self.logger.debug(f"========= not terminal: {node.name}")
                node = node.expand()
                self.tree.add_node(node)
                path.append(node)

                # self.logger.debug(f"========= expanded:{n.name for n in node}")
                reward = node.simulate(path)
            else:
                self.logger.debug(f"========= terminal:{node.name}")
                if path not in paths:
                    paths.append(path)
                reward = self.game.get_reward(path)

            node.backprop(reward)
        return self.get_best_path()

    def get_best_path(
        self
    ) -> List[NodeBase]:

        node = self.tree.get_root()
        path = [node]
        while len(node.children) > 0:
            node = node.select()
            path.append(node)
        return path



    def draw_tree(self, out_path: str, plot: bool = False, *args) -> None:
        self.tree.draw_tree(out_name=out_path, plot=plot, *args)

    def shut_down(
        self,
        mcts_path: str = None,
        tree_path: str = None
    ) -> None:

        self.logger.debug("Shutting down SP-MCTS")
        if mcts_path:
            with open(mcts_path, "wb") as f:
                pickle.dump(self, f)
        if tree_path:
            self.tree.save(tree_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    set_seed(250)
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

    tree_grammar = {'start': 'CLF', 'non_terminals': ['Input', 'CLF', 'DataPrep', 'NumPrep', 'Imputer'],
                    'terminals': ['csv', 'random_forest', 's_imp', 'pca', 'minmax', 'decision_tree'],
                    'rules': {'Input': {'csv': []},
                              'CLF': {'random_forest': ['DataPrep', 'Input'], 'decision_tree': ['DataPrep', 'Input']},
                              'NumPrep': {'s_imp': ['Input'], 'minmax': ['Imputer']},
                              'DataPrep': {'s_imp': ['Input'], 'pca': ['NumPrep', 'Input'], 'minmax': ['Imputer']},
                              'Imputer': {'s_imp': ['Input']}}}
    #
    # tree_grammar = {
    #     "start": "A",
    #     "terminals": ["c", "d", "e"],
    #     "non_terminals": ["A", "D", "E"],
    #     "rules": {
    #         "A": {"c": ["D", "E"], "d": []},
    #         "D": {"d": []},
    #         "E": {"e": ["A"]}
    #     }
    # }

    hypergraph_dict = get_hypergraph_dict_from_tree_grammar(tree_grammar)
    hypergraph = build_hypergraph(hypergraph_dict)
    plot_hypergraph_components(hypergraph, "hypergraph.png", start_node=tree_grammar["start"], node_size=5000,
                               node_font_size=11)

    params = {
        "num_iterations": 100,
        "exploration_param": 0.5,
        "num_simulations": 2,
    }

    # evaluator = Evaluator()
    game = HyperGraphGame(hypergraph)

    mcts = PureSinglePlayerMCTS(
        game=game,
        parameters=params,
        selection_policy=UCT,
    )
    best_path = mcts.run()


    mcts.draw_tree("nx_di_graph.png", plot=True)
    mcts.shut_down("mcts.pkl", "nx_di_graph.pkl")

    print("start", tree_grammar["start"])
    print("non_terminals", tree_grammar["non_terminals"])
    print("terminals", tree_grammar["terminals"])
    print("rules")
    for k, v in tree_grammar["rules"].items():
        print(k)
        print(v)
        print()
