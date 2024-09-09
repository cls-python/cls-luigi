from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from typing import Dict, Type, Any, List

if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node
    from cls_luigi.search.core.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy
    from cls_luigi.search.core.tree import TreeBase
    from cls_luigi.search.mcts.game import OnePlayerGame

import logging



from cls_luigi.search.core.mcts import SinglePlayerMCTS
from cls_luigi.search.mcts.tree import MCTSTreeWithGrammar
from cls_luigi.search.mcts.node import NodeFactory
from cls_luigi.search.mcts.policy import UCT, RandomExpansion, RandomSimulation


class RecursiveSinglePlayerMCTS(SinglePlayerMCTS):
    """Recursive implementation of a Single player Monte Carlo Tree Search ."""

    def __init__(
        self,
        parameters: Dict[str, Any],
        game: OnePlayerGame,
        selection_policy: Type[SelectionPolicy] = UCT,
        expansion_policy: Type[ExpansionPolicy] = RandomExpansion,
        tree_cls: Type[TreeBase] = MCTSTreeWithGrammar,
        node_factory_cls: NodeFactory = NodeFactory,
        fully_expanded_params: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:

        super().__init__(
            parameters=parameters,
            game=game,
            selection_policy=selection_policy,
            expansion_policy=expansion_policy,
            tree_cls=tree_cls,
            node_factory_cls=node_factory_cls,
            fully_expanded_params=fully_expanded_params,
            logger=logger)

    def run(
        self
    ) -> List[Node]:
        self.logger.debug("Running SP-MCTS for {} iterations".format(self.parameters["num_iterations"]))

        for iter_ix in range(self.parameters["num_iterations"]):
            path = []
            logging.debug("Iteration: {}".format(iter_ix))
            node = self.tree.get_root()
            path.append(node)

            while node.is_fully_expanded() and not self.game.is_final_state(node):
                self.logger.debug(f"fully expanded: {node.name}")
                node = node.select()
                if node:
                    path.append(node)
                    self.logger.debug(f"selected new node: {node.name}")
                else:
                    self.logger.debug(f"Breaking selection loop. Node is None.")
                    break
            if node:
                while not self.game.is_final_state(node):
                    self.logger.debug(f"Expanding {node.name}")
                    node = node.expand()
                    self.tree.add_node(node)
                    path.append(node)

                reward = self.game.get_reward(path)
                self._update_incumbent(path, reward)
                node.backprop(reward)
                self.logger.debug(f"==================================\n==================================\n\n\n")

                # self.draw_tree(f"/home/hadi/Documents/cls-luigi/examples/ml_blood_sugar_level/mcts_imgs/nx_di_graph_iter{iter_ix}.png", plot=False)

        # print("N evaluated pipelines (unique):", len(self.game.evaluator.evaluated))
        # print("N failed (unique):", len(self.game.evaluator.failed))
        # print("N not found paths (unique):", len(self.game.evaluator.not_found_paths))
        return self.get_incumbent()


if __name__ == "__main__":
    from cls_luigi.search.helpers import set_seed
    from cls_luigi.grammar.hypergraph import get_hypergraph_dict_from_tree_grammar, plot_hypergraph_components, \
        build_hypergraph
    from cls_luigi.search.mcts.game import HyperGraphGame

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
    #
    # tree_grammar = {'start': 'CLF', 'non_terminals': ['Input', 'CLF', 'DataPrep', 'NumPrep', 'Imputer'],
    #                 'terminals': ['csv', 'random_forest', 's_imp', 'pca', 'minmax', 'decision_tree',
    #                               "adaboost", "bernollinb", "extra_trees", "gradient_boosting", "gaussiannb",
    #                               "knn", "ida", "lin_svc", "mlp", "multinb", "passive_aggressive", "sgd", "svc",
    #                               "s_imp", "minmax", "robust", "power", "quantile", "standard", "pca", "ica",
    #                               "feat_ag", "k_pca", "nystroem", "poly", "rt_embedd", "rfb", "select_ext_t",
    #                               "select_svc", "select_percent", "select_rates"
    #
    #                               ],
    #                 'rules': {'Input': {'csv': []},
    #                           'CLF': {
    #                               'random_forest': ['DataPrep', 'Input'],
    #                               'decision_tree': ['DataPrep', 'Input'],
    #                               'adaboost': ['DataPrep', 'Input'],
    #                               'bernollinb': ['DataPrep', 'Input'],
    #                               'extra_trees': ['DataPrep', 'Input'],
    #                               'gradient_boosting': ['DataPrep', 'Input'],
    #                               'gaussiannb': ['DataPrep', 'Input'],
    #                               'knn': ['DataPrep', 'Input'],
    #                               'ida': ['DataPrep', 'Input'],
    #                               'lin_svc': ['DataPrep', 'Input'],
    #                               'mlp': ['DataPrep', 'Input'],
    #                               'multinb': ['DataPrep', 'Input'],
    #                               'passive_aggressive': ['DataPrep', 'Input'],
    #                               'sgd': ['DataPrep', 'Input'],
    #                               'svc': ['DataPrep', 'Input'],
    #
    #                           },
    #                           'NumPrep': {'s_imp': ['Input'],
    #                                       'minmax': ['Imputer'],
    #                                       'robust': ['Imputer'],
    #                                       'power': ['Imputer'],
    #                                       'quantile': ['Imputer'],
    #                                       'standard': ['Imputer'],
    #
    #                                       },
    #                           'DataPrep': {'s_imp': ['Input'],
    #
    #                                        'pca': ['NumPrep', 'Input'],
    #                                        "ica": ['NumPrep', 'Input'],
    #                                        'feat_ag': ['NumPrep', 'Input'],
    #                                        "k_pca": ['NumPrep', 'Input'],
    #                                        'nystroem': ['NumPrep', 'Input'],
    #                                        "poly": ['NumPrep', 'Input'],
    #                                        'rt_embedd': ['NumPrep', 'Input'],
    #                                        "rfb": ['NumPrep', 'Input'],
    #
    #                                        "select_ext_t": ['NumPrep', 'Input'],
    #                                        'select_svc': ['NumPrep', 'Input'],
    #                                        "select_percent": ['NumPrep', 'Input'],
    #                                        "select_rates": ['NumPrep', 'Input'],
    #
    #                                        'minmax': ['Imputer'],
    #                                        'robust': ['Imputer'],
    #                                        'power': ['Imputer'],
    #                                        'quantile': ['Imputer'],
    #                                        'standard': ['Imputer'],
    #
    #                                        },
    #                           'Imputer': {'s_imp': ['Input']}}}
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
    plot_hypergraph_components(hypergraph, "hypergraph.png", node_size=5000,
                               node_font_size=11)

    params = {
        "num_iterations": 50,
        "exploration_param": 0.5,
        # "num_simulations": 2,
    }

    game = HyperGraphGame(hypergraph, minimization_problem=True)

    # progressive_widening_params = {
    #     "threshold": 2,
    #     "progressiv_widening_coeff": 0.5,
    #     "max_children": 10
    # }

    mcts = RecursiveSinglePlayerMCTS(
        game=game,
        parameters=params,
        selection_policy=UCT,
        # fully_expanded_params=progressive_widening_params,
    )
    best_path = mcts.run()

    mcts.draw_tree("nx_di_graph.png", plot=True)
    # mcts.shut_down("mcts.pkl", "nx_di_graph.pkl")

    print("start", tree_grammar["start"])
    print("non_terminals", tree_grammar["non_terminals"])
    print("terminals", tree_grammar["terminals"])
    print("rules")
    for k, v in tree_grammar["rules"].items():
        print(k)
        print(v)
        print()
