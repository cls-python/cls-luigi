import json
import logging
import pickle
from typing import Dict, List, Tuple

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.core.tree import TreeBase


class MCTSTreeWithGrammar(TreeBase):
    def __init__(
        self,
        root: NodeBase,
        grammar: Dict[str, Dict[str, List[str]] | str | List[str]],
        logger: logging.Logger = None,
        **kwargs
    ) -> None:
        super().__init__(root, **kwargs)

        self.G = nx.DiGraph()
        self.id_count = -1
        self.add_root(self.root)
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
        non_terminal_nodes_color: str = "#003049",
        terminal_nodes_color: str = "#b87012",
        choice_edges_color: str = "#818589",
        choice_edges_style: str = "dotted",
        arrow_style: str = "->",
        arrows_size: int = 28,
        arrow_width: int = 2,
        min_target_margin: int = 25,
        legend_loc: str = 'best',
        out_dpi: int = 600,
        plot: bool = False

    ) -> None:

        fig, axs = plt.subplots(1, 1, figsize=figsize)
        axs.axis('off')
        fig.patch.set_facecolor(facecolor)

        pos = nx.bfs_layout(self.G, start=start_node_id)

        non_terminal_nodes = [node_id for node_id in self.G.nodes if
                              self.get_node(node_id).name in self.grammar["non_terminals"]]
        terminal_nodes = [node_id for node_id in self.G.nodes if
                          self.get_node(node_id).name in self.grammar["terminals"]]

        nx.draw_networkx_nodes(self.G,
                               pos,
                               node_size=node_size,
                               ax=axs,
                               node_color=non_terminal_nodes_color,
                               node_shape='s',
                               nodelist=non_terminal_nodes)

        nx.draw_networkx_nodes(self.G,
                               pos,
                               node_size=node_size,
                               ax=axs,
                               node_color=terminal_nodes_color,
                               node_shape='o',
                               nodelist=terminal_nodes)

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
            q = round(self.get_node(target).reward / self.get_node(target).visits, 2)
            edge_labels[(source, target)] = f"Q={q}"

        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, ax=axs)

        legend_elements = [
            Patch(facecolor=non_terminal_nodes_color, edgecolor=non_terminal_nodes_color, label="Non-terminals"),
            Line2D([0], [0], marker='o', color=terminal_nodes_color,
                   label='Terminals', markerfacecolor=terminal_nodes_color, markersize=14, linewidth=0),
        ]

        axs.legend(handles=legend_elements, loc=legend_loc)

        plt.tight_layout()
        if out_name:
            plt.savefig(out_name, dpi=out_dpi)

        if plot:
            plt.show()

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
