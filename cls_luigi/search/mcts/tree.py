import logging
import math
import pickle
from typing import Tuple, Literal, Type

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.core.tree import TreeBase

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class MCTSTreeWithGrammar(TreeBase):
    def __init__(
        self,
        root: Type[NodeBase],
        hypergraph: nx.Graph,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:
        super().__init__(root, logger, **kwargs)

        self.G = nx.DiGraph()
        self.hypergraph = hypergraph
        self.id_count = -1
        self.add_root(self.root)
        # self.grammar = grammar

    def add_root(
        self,
        node: Type[NodeBase]
    ) -> None:

        self.id_count += 1
        node.node_id = self.id_count
        self.G.add_node(node.node_id, value=node,
                        leaf_node=node.game.is_final_state(node.name),
                        start_node=node.game.is_start(node.name[0])
                        )
        self.logger.debug(f"Added root node {node.name}")

    def add_node(
        self,
        node: NodeBase
    ) -> None:

        self.id_count += 1
        node.node_id = self.id_count
        self.G.add_node(node.node_id, value=node,
                        leaf_node=node.game.is_final_state(node.name),
                        start_node=node.game.is_start(node.name[0]))
        self.G.add_edge(node.parent.node_id, node.node_id)
        self.logger.debug(f"Added node {node.name} and an edge from {node.parent.name}")

    def get_node(
        self,
        node_id: int
    ) -> NodeBase:

        return self.G.nodes[node_id]["value"]

    def get_root(
        self
    ) -> NodeBase:

        return self.get_node(0)

    @staticmethod
    def _scale_figure_size(num_nodes, base_size=50, scale_factor=0.6):

        scaled_size = base_size + scale_factor * num_nodes**0.5
        return (scaled_size, scaled_size/1.61)

    def draw_tree(
        self,
        out_name: str = None,
        start_node_id: int = 0,
        node_font_size: int = 10,
        node_size: int = 800,
        figsize: Tuple[int, int] = None,
        facecolor: str = "White",
        non_terminal_nodes_color: str = "#003049",
        terminal_nodes_color: str = "#b87012",
        leaf_node_color: str = "#94505c",
        start_node_color: str = "#556B2F",
        choice_edges_color: str = "#818589",
        choice_edges_style: str = "dotted",
        arrow_style: str = "->",
        arrows_size: int = 28,
        arrow_width: int = 2,
        min_target_margin: int = 25,
        legend_loc: str = 'best',
        out_dpi: int = 600,
        plot: bool = False,
        plot_title: str = "MCTS Tree",
        non_terminal_node_shape: str = 's',
        terminal_node_shape: str = 'o',
        node_label_color="white",
        title_fontsize: int = 15,
        title_font_weight: str = 'bold',
        title_loc: Literal['center', "left", "right"] = "center"

    ) -> None:

        if not figsize:
            figsize = self._scale_figure_size(len(self.G.nodes))

        fig, axs = plt.subplots(1, 1, figsize=figsize)
        axs.axis('off')
        fig.patch.set_facecolor(facecolor)

        pos = nx.bfs_layout(self.G, start=start_node_id)

        non_terminal_nodes = []
        terminal_nodes = []
        leaf_node = []
        start_node = []

        for node in self.G.nodes(data=True):
            term = node[1]["value"].name[0]
            if self.hypergraph.nodes[term]["terminal_node"]:
                terminal_nodes.append(node[0])
            else:
                non_terminal_nodes.append(node[0])

            if node[1]["leaf_node"]:
                leaf_node.append(node[0])
            elif node[1]["start_node"]:
                start_node.append(node[0])


        nx.draw_networkx_nodes(self.G,
                               pos,
                               node_size=node_size * 2,
                               ax=axs,
                               node_color=non_terminal_nodes_color,
                               node_shape=non_terminal_node_shape,
                               nodelist=non_terminal_nodes)

        nx.draw_networkx_nodes(self.G,
                               pos,
                               node_size=node_size * 4,
                               ax=axs,
                               node_color=terminal_nodes_color,
                               node_shape=terminal_node_shape,
                               nodelist=terminal_nodes)


        nx.draw_networkx_nodes(self.G,
                               pos,
                               node_size=node_size * 4,
                               ax=axs,
                               node_color=leaf_node_color,
                               node_shape=terminal_node_shape,
                               nodelist=leaf_node)

        nx.draw_networkx_nodes(self.G,
                               pos,
                               node_size=node_size * 4,
                               ax=axs,
                               node_color=start_node_color,
                               node_shape=non_terminal_node_shape,
                               nodelist=start_node)

        nx.draw_networkx_edges(self.G, pos, ax=axs, edge_color=choice_edges_color,
                               arrowstyle=arrow_style,
                               arrowsize=arrows_size,
                               width=arrow_width,
                               style=choice_edges_style,
                               min_target_margin=min_target_margin)

        labels = {i: self.G.nodes[i]["value"].name for i in self.G.nodes}
        for k, v in labels.items():
            if isinstance(v, tuple):
                new_label = ""
                for j in v:
                    new_label += j
                    if len(v) > 1:
                        new_label += "\n"
                labels[k] = new_label

        nx.draw_networkx_labels(self.G, pos, labels,
                                ax=axs, font_color=node_label_color, font_size=node_font_size)

        edge_labels = {}

        for edge in list(self.G.edges):
            source, target = edge
            q = round(self.get_node(target).reward / self.get_node(target).visits, 2)
            edge_labels[(source, target)] = f"Q={q}\nV={self.get_node(target).visits}"

        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, ax=axs)

        legend_elements = [
            Patch(facecolor=non_terminal_nodes_color, edgecolor=non_terminal_nodes_color, label="Non-terminals"),
            Patch(facecolor=start_node_color, edgecolor=start_node_color, label="Start"),

            Line2D([0], [0], marker=terminal_node_shape, color=terminal_nodes_color,
                   label='Terminals', markerfacecolor=terminal_nodes_color, markersize=14, linewidth=0),

            Line2D([0], [0], marker=terminal_node_shape, color=leaf_node_color,
                   label='Leaf', markerfacecolor=leaf_node_color, markersize=14, linewidth=0),
        ]

        axs.set_title(plot_title, fontsize=title_fontsize, fontweight=title_font_weight, loc=title_loc)
        axs.legend(handles=legend_elements, loc=legend_loc)

        plt.tight_layout()
        if out_name:
            plt.savefig(out_name, dpi=out_dpi)

        if plot:
            plt.show()

        plt.close()

    def save(
        self,
        path: str
    ):
        if not path.endswith(".pkl"):
            self.logger.debug("Appending .pkl to the path")
            path = path + ".pkl"

        # Save the graph to a JSON file
        with open(path, 'wb') as f:
            pickle.dump(self.G, f)
