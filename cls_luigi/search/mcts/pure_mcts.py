import logging
from typing import Dict, Type, Any, List

from cls_luigi.grammar.hypergraph import get_hypergraph_dict_from_tree_grammar, plot_hypergraph_components, \
    build_hypergraph
from cls_luigi.search.core.mcts import SinglePlayerMCTS
from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.core.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy
from cls_luigi.search.core.tree import TreeBase
from cls_luigi.search.helpers import set_seed
from cls_luigi.search.mcts.game import HyperGraphGame, OnePlayerGame
from cls_luigi.search.mcts.tree import MCTSTreeWithGrammar
from cls_luigi.search.mcts.node import NodeFactory
from cls_luigi.search.mcts.policy import UCT, RandomExpansion, RandomSimulation


class PureSinglePlayerMCTS(SinglePlayerMCTS):
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

        super().__init__(
            parameters=parameters,
            game=game,
            selection_policy=selection_policy,
            expansion_policy=expansion_policy,
            simulation_policy=simulation_policy,
            tree_cls=tree_cls,
            node_factory_cls=node_factory_cls,
            logger=logger)

    def run(
        self
    ) -> List[NodeBase]:
        self.logger.debug("Running SP-MCTS for {} iterations".format(self.parameters["num_iterations"]))

        paths = []

        for _ in range(self.parameters["num_iterations"]):
            path = []
            logging.debug("======== Iteration: {}".format(_))
            node = self.tree.get_root()
            path.append(node)

            while node.is_fully_expanded() and not self.game.is_final_state(node):
                self.logger.debug(f"========= fully expanded: {node.name}")
                node = node.select()
                path.append(node)
                self.logger.debug(f"========= selected new node: {node.name}")

            if not self.game.is_final_state(node):
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
            if self.game.is_final_state(node):
                self._update_incumbent(path, reward)
            node.backprop(reward)
            # self.draw_tree(plot=True)

        return self.get_incumbent()


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

    tree_grammar = {'start': 'CLF', 'non_terminals': ['Input', 'CLF', 'DataPrep', 'NumPrep', 'Imputer'],
                    'terminals': ['csv', 'random_forest', 's_imp', 'pca', 'minmax', 'decision_tree',
                                  "adaboost", "bernollinb", "extra_trees", "gradient_boosting", "gaussiannb",
                                  "knn", "ida", "lin_svc", "mlp", "multinb", "passive_aggressive", "sgd", "svc",
                                  "s_imp", "minmax", "robust", "power", "quantile", "standard", "pca", "ica",
                                  "feat_ag", "k_pca", "nystroem", "poly", "rt_embedd", "rfb", "select_ext_t",
                                  "select_svc", "select_percent", "select_rates"

                                  ],
                    'rules': {'Input': {'csv': []},
                              'CLF': {
                                  'random_forest': ['DataPrep', 'Input'],
                                  'decision_tree': ['DataPrep', 'Input'],
                                  'adaboost': ['DataPrep', 'Input'],
                                  'bernollinb': ['DataPrep', 'Input'],
                                  'extra_trees': ['DataPrep', 'Input'],
                                  'gradient_boosting': ['DataPrep', 'Input'],
                                  'gaussiannb': ['DataPrep', 'Input'],
                                  'knn': ['DataPrep', 'Input'],
                                  'ida': ['DataPrep', 'Input'],
                                  'lin_svc': ['DataPrep', 'Input'],
                                  'mlp': ['DataPrep', 'Input'],
                                  'multinb': ['DataPrep', 'Input'],
                                  'passive_aggressive': ['DataPrep', 'Input'],
                                  'sgd': ['DataPrep', 'Input'],
                                  'svc': ['DataPrep', 'Input'],

                              },
                              'NumPrep': {'s_imp': ['Input'],
                                          'minmax': ['Imputer'],
                                          'robust': ['Imputer'],
                                          'power': ['Imputer'],
                                          'quantile': ['Imputer'],
                                          'standard': ['Imputer'],

                                          },
                              'DataPrep': {'s_imp': ['Input'],

                                           'pca': ['NumPrep', 'Input'],
                                           "ica": ['NumPrep', 'Input'],
                                           'feat_ag': ['NumPrep', 'Input'],
                                           "k_pca": ['NumPrep', 'Input'],
                                           'nystroem': ['NumPrep', 'Input'],
                                           "poly": ['NumPrep', 'Input'],
                                           'rt_embedd': ['NumPrep', 'Input'],
                                           "rfb": ['NumPrep', 'Input'],

                                           "select_ext_t": ['NumPrep', 'Input'],
                                           'select_svc': ['NumPrep', 'Input'],
                                           "select_percent": ['NumPrep', 'Input'],
                                           "select_rates": ['NumPrep', 'Input'],

                                           'minmax': ['Imputer'],
                                           'robust': ['Imputer'],
                                           'power': ['Imputer'],
                                           'quantile': ['Imputer'],
                                           'standard': ['Imputer'],

                                           },
                              'Imputer': {'s_imp': ['Input']}}}

    hypergraph_dict = get_hypergraph_dict_from_tree_grammar(tree_grammar)
    hypergraph = build_hypergraph(hypergraph_dict)
    plot_hypergraph_components(hypergraph, "hypergraph.png", start_node=tree_grammar["start"], node_size=5000,
                               node_font_size=11)

    params = {
        "num_iterations": 5,
        "exploration_param": 0.5,
        "num_simulations": 2,
    }

    # evaluator = Evaluator()
    game = HyperGraphGame(hypergraph, minimization_problem=True)

    mcts = PureSinglePlayerMCTS(
        game=game,
        parameters=params,
        selection_policy=UCT,
    )
    best_path = mcts.run()

    mcts.draw_tree("nx_di_graph.png", plot=True)
    mcts.shut_down("mcts.pkl", "nx_di_graph.pkl")

