import logging

import networkx as nx
from typing import Tuple, Dict, List, Type, Any
from matplotlib import pyplot as plt

from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.core.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy
from cls_luigi.search.mcts.game import TreeGrammarGame, OnePlayerGame
from cls_luigi.search.mcts.node import Node
from cls_luigi.search.mcts.policy import UCB1, RandomExpansion, RandomSimulation


class Tree:
    def __init__(
        self,
        root: NodeBase,
        grammar: Dict[str, Dict[str, List[str]] | str | List[str]],
        logger: logging.Logger = None

    ) -> None:

        self.G = nx.DiGraph()
        self.id_count = -1
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.add_root(root)
        self.grammar = grammar


    def add_root(
        self,
        node: NodeBase
    ) -> None:

        self.id_count += 1
        node.node_id = self.id_count
        self.G.add_node(node.node_id, value=node)
        self.logger.debug(f"Added root node {node.name}")

    def add_node(
        self,
        node: NodeBase
    ) -> None:

        self.id_count += 1
        node.node_id = self.id_count
        self.G.add_node(node.node_id, value=node)
        self.G.add_edge(node.parent.node_id, node.node_id)
        self.logger.debug(f"Added node {node.name} and an edge from {node.parent.name}")

    def draw(
        self
    ) -> None:

        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, font_weight='bold')

        labels = {i: self.G.nodes[i]["value"].name for i in self.G.nodes}
        nx.draw_networkx_labels(self.G, pos, labels)
        plt.show()

    def get_node(
        self,
        node_id: int
    ) -> NodeBase:

        return self.G.nodes[node_id]["value"]

    def get_root(
        self
    ) -> NodeBase:

        return self.get_node(0)

    def draw_tree(
        self,
        out_name: str = None,
        start_node_id: int = 0,
        node_font_size: int = 10,
        node_size: int = 2000,
        figsize: Tuple[int, int] = (22, 16),
        facecolor: str = "White",
        type_nodes_color: str = "#003049",
        subtype_nodes_color: str = "#b87012",
        choice_edges_color: str = "#818589",
        arg_edges_color: str = "black",
        choice_edges_style: str = "dotted",
        arg_edges_style: str = "solid",
        arrow_style: str = "->",
        arrows_size: int = 28,
        arrow_width: int = 2,
        min_target_margin: int = 25,
        legend_loc: str = 'best',
        out_dpi: int = 600
    ) -> None:

        fig, axs = plt.subplots(1, 1, figsize=figsize)
        axs.axis('off')
        fig.patch.set_facecolor(facecolor)

        pos = nx.bfs_layout(self.G, start=start_node_id)

        nx.draw_networkx_nodes(self.G, pos, node_size=node_size,
                               ax=axs, node_color=type_nodes_color, node_shape='s')

        nx.draw_networkx_edges(self.G, pos, ax=axs, edge_color=choice_edges_color,
                               arrowstyle=arrow_style,
                               arrowsize=arrows_size,
                               width=arrow_width,
                               style=choice_edges_style,
                               min_target_margin=min_target_margin)

        labels = {i: self.G.nodes[i]["value"].name for i in self.G.nodes}

        nx.draw_networkx_labels(self.G, pos, labels,
                                ax=axs, font_color='white', font_size=node_font_size)

        edge_labels = {}

        for edge in list(self.G.edges):
            source, target = edge
            reward = round(self.get_node(target).reward, 2)
            # edge_labels[(source, target)] = f"v={self.get_node(target).visits}, r={reward}"
            edge_labels[(source, target)] = f"r={reward}"

        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, ax=axs)

        plt.tight_layout()
        if out_name:
            plt.savefig(out_name, dpi=out_dpi)
        plt.show()


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
        logger: logging.Logger = None
    ) -> None:

        self.grammar = grammar
        self.game = game_class(self.grammar)
        self.parameters = parameters
        self.selection_policy = selection_policy
        self.expansion_policy = expansion_policy
        self.simulation_policy = simulation_policy
        self.tree = Tree(root=self.init_root(), grammar=grammar)
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
        # self.tree.draw_tree()
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
    logging.basicConfig(level=logging.INFO)
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
        "num_iterations": 1000,
        "exploration_param": 0.5
    }

    mcts = SP_MCTS(
        game_class=TreeGrammarGame,
        grammar=tree_grammar,
        parameters=params,
        selection_policy=UCB1,
    )
    mcts.run()
