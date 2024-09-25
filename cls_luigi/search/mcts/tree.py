from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List
from typing import Tuple, Literal

from cls_luigi.tools.io_functions import dump_pickle

if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node

import logging
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from cls_luigi.search.core.tree import TreeBase
from os.path import join as pjoin

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class MCTSTreeWithGrammar(TreeBase):
    def __init__(
        self,
        root: Node,
        hypergraph: nx.Graph,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> None:
        super().__init__(root, logger, **kwargs)

        self.G = nx.DiGraph()
        self.hypergraph = hypergraph
        self.id_count = -1  # To start from 0
        self.add_root(self.root)

    def add_root(
        self,
        node: Node
    ) -> None:

        self.id_count += 1
        node.node_id = self.id_count
        self.G.add_node(node.node_id, value=node,
                        leaf_node=node.game.is_final_state(node),
                        start_node=node.game.is_start(node.name[0])
                        )
        self.logger.debug(f"Added root node {node.name}")

    def add_node(
        self,
        node: Node
    ) -> None:

        self.id_count += 1
        node.node_id = self.id_count
        self.G.add_node(node.node_id, value=node,
                        leaf_node=node.game.is_final_state(node),
                        start_node=node.game.is_start(node.name[0]))
        self.G.add_edge(node.parent.node_id, node.node_id)
        self.logger.debug(f"Added node {node.name} and an edge from {node.parent.name}")

    def get_node(
        self,
        node_id: int
    ) -> Node:

        return self.G.nodes[node_id]["value"]

    def get_root(
        self,
        node_id: int = 0
    ) -> Node:
        self.logger.debug("Returning root node")
        return self.get_node(node_id)

    @staticmethod
    def _scale_figure_size(num_nodes, base_size=20, scale_factor=0.6):

        scaled_size = base_size + scale_factor * num_nodes ** 0.5
        return scaled_size, scaled_size / 1.61

    def _get_structured_node_labels(
        self,
    ) -> dict[int, str]:

        labels = {i: self.G.nodes[i]["value"].name for i in self.G.nodes}
        for k, v in labels.items():
            if isinstance(v, tuple):
                new_label = ""
                for ix, j in enumerate(v):
                    new_label += j
                    if len(v) > 1 and ix < len(v) - 1:
                        new_label += "\n"
                labels[k] = new_label
        return labels

    def _get_structured_edge_labels(
        self
    ) -> dict[tuple[int, int], str]:

        edge_labels = {}
        for edge in list(self.G.edges):
            source, target = edge
            q = round(self.get_node(target).sum_rewards / self.get_node(target).visits, 2)
            edge_labels[(source, target)] = f"Q={q}\nV={self.get_node(target).visits}"
        return edge_labels

    def render(
        self,
        best_mcts_path: Optional[List[Node]] = None,
        out_path: Optional[str] = None,
        start_node_id: int = 0,
        node_font_size: int = 10,
        node_size: int = 800,
        figsize: Optional[Tuple[int, int]] = None,
        facecolor: str = "White",
        non_terminal_nodes_color: str = "#003049",
        terminal_nodes_color: str = "#b87012",
        incumbent_node_edge_color: str = "red",
        leaf_node_color: str = "#94505c",
        start_node_color: str = "#556B2F",
        choice_edges_color: str = "#818589",
        choice_edges_style: str = "dotted",
        arrow_style: str = "->",
        arrows_size: int = 28,
        arrow_width: int = 2,
        min_target_margin: int = 25,
        legend_loc: str = 'best',
        out_dpi: int = 300,
        show: bool = False,
        plot_title: str = "MCTS Tree",
        non_terminal_node_shape: str = 's',
        terminal_node_shape: str = 'o',
        node_label_color="white",
        title_fontsize: int = 30,
        title_font_weight: str = 'bold',
        title_loc: Literal['center', "left", "right"] = "center",
        node_line_widths: float | int = 3
    ) -> None:

        if not figsize:
            figsize = self._scale_figure_size(len(self.G.nodes))

        fig, axs = plt.subplots(1, 1, figsize=figsize)
        axs.axis('off')
        fig.patch.set_facecolor(facecolor)

        pos = nx.bfs_layout(self.G, start=start_node_id)

        incumbent_node_ids = [node.node_id for node in best_mcts_path]

        terminal_nodes = []
        terminal_node_sizes = []
        terminal_node_colors = []
        terminal_node_boarder_colors = []

        non_terminal_nodes = []
        non_terminal_node_sizes = []
        non_terminal_node_colors = []
        non_terminal_node_boarder_colors = []

        for node in self.G.nodes(data=True):
            term = node[1]["value"].name[0]
            node_id = node[0]

            if node[1]["leaf_node"]:
                terminal_nodes.append(node_id)
                terminal_node_sizes.append(node_size * 4)
                terminal_node_colors.append(leaf_node_color)

                if node_id in incumbent_node_ids:
                    terminal_node_boarder_colors.append(incumbent_node_edge_color)
                else:
                    terminal_node_boarder_colors.append(leaf_node_color)

            elif node[1]["start_node"]:
                non_terminal_nodes.append(node_id)
                non_terminal_node_sizes.append(node_size * 2)
                non_terminal_node_colors.append(start_node_color)
                non_terminal_node_boarder_colors.append(incumbent_node_edge_color)

            elif self.hypergraph.nodes[term]["terminal_node"]:
                terminal_nodes.append(node_id)
                terminal_node_sizes.append(node_size * 4)
                terminal_node_colors.append(terminal_nodes_color)
                if node_id in incumbent_node_ids:
                    terminal_node_boarder_colors.append(incumbent_node_edge_color)
                else:
                    terminal_node_boarder_colors.append(terminal_nodes_color)

            elif not self.hypergraph.nodes[term]["terminal_node"]:
                non_terminal_nodes.append(node_id)
                non_terminal_node_sizes.append(node_size * 2)
                non_terminal_node_colors.append(non_terminal_nodes_color)
                if node_id in incumbent_node_ids:
                    non_terminal_node_boarder_colors.append(incumbent_node_edge_color)
                else:
                    non_terminal_node_boarder_colors.append(non_terminal_nodes_color)

        nx.draw_networkx_nodes(self.G,
                               pos,
                               node_size=non_terminal_node_sizes,
                               ax=axs,
                               node_color=non_terminal_node_colors,
                               node_shape=non_terminal_node_shape,
                               nodelist=non_terminal_nodes,
                               edgecolors=non_terminal_node_boarder_colors,
                               linewidths=node_line_widths)

        nx.draw_networkx_nodes(self.G,
                               pos,
                               node_size=terminal_node_sizes,
                               ax=axs,
                               node_color=terminal_node_colors,
                               node_shape=terminal_node_shape,
                               nodelist=terminal_nodes,
                               edgecolors=terminal_node_boarder_colors,
                               linewidths=node_line_widths)

        nx.draw_networkx_edges(self.G, pos, ax=axs, edge_color=choice_edges_color,
                               arrowstyle=arrow_style,
                               arrowsize=arrows_size,
                               width=arrow_width,
                               style=choice_edges_style,
                               min_target_margin=min_target_margin)

        labels = self._get_structured_node_labels()
        nx.draw_networkx_labels(self.G, pos, labels,
                                ax=axs, font_color=node_label_color, font_size=node_font_size)

        edge_labels = self._get_structured_edge_labels()
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, ax=axs)

        legend_elements = [
            Patch(facecolor=non_terminal_nodes_color, edgecolor=non_terminal_nodes_color, label="Non-terminals"),
            Patch(facecolor=start_node_color, edgecolor=start_node_color, label="Start"),

            Line2D([0], [0], marker=terminal_node_shape, color=terminal_nodes_color,
                   label='Terminals', markerfacecolor=terminal_nodes_color, markersize=14, linewidth=0),

            Line2D([0], [0], marker=terminal_node_shape, color=leaf_node_color,
                   label='Leaf', markerfacecolor=leaf_node_color, markersize=14, linewidth=0),
        ]
        plot_title += "\n(Incumbent path is highlighted in red)"
        axs.set_title(plot_title, fontsize=title_fontsize, fontweight=title_font_weight, loc=title_loc)

        axs.legend(handles=legend_elements, loc=legend_loc)

        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=out_dpi)

        if show:
            plt.show()

        plt.close()

    def save(
        self,
        out_path: str,
    ) -> None:
        dump_pickle(self.G, out_path)
