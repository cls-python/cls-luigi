from typing import Dict, List, Tuple

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def get_hypergraph_dict_from_tree_grammar(
    tree_grammar: Dict[str, Dict[str, List[str]]]
) -> Dict[str, List[str]]:

    hypergraph = {
        "choice_edges": [],
        "arg_edges": [],
        "non_terminal_nodes": [],
        "terminal_nodes": []}

    for lhs, rhs in tree_grammar.items():

        if lhs not in hypergraph["non_terminal_nodes"]:
            hypergraph["non_terminal_nodes"].append(lhs)

        for combinator, args in rhs.items():

            if combinator not in hypergraph["terminal_nodes"]:
                hypergraph["terminal_nodes"].append(combinator)

            hypergraph["choice_edges"].append((lhs, combinator))
            for arg in args:
                if arg not in hypergraph["non_terminal_nodes"]:
                    hypergraph["non_terminal_nodes"].append(arg)

                hypergraph["arg_edges"].append((combinator, arg))

    return hypergraph


def plot_hypergraph_components(
    hypergraph: Dict[str, List[str]],
    out_name: str,
    start_node: str,
    node_font_size: int = 5,
    node_size: int = 5000,
    figsize: Tuple[int, int] = (13, 8),
    facecolor: str = "White",
    non_terminal_nodes_color: str = "#003049",
    terminal_nodes_color: str = "#b87012",
    choice_edges_color: str = "#818589",
    arg_edges_color: str = "black",
    choice_edges_style: str = "dotted",
    arg_edges_style: str = "solid",
    arrow_style: str = "->",
    arrows_size: int = 28,
    arrow_width: int = 2,
    min_target_margin: int = 38,
    legend_loc: str = 'best',
    out_dpi: int = 600
) -> None:

    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.axis('off')
    fig.patch.set_facecolor(facecolor)

    all_edges = hypergraph["choice_edges"] + hypergraph["arg_edges"]
    all_nodes = hypergraph["non_terminal_nodes"] + hypergraph["terminal_nodes"]

    g = nx.DiGraph()
    g.add_nodes_from(all_nodes)
    g.add_edges_from(all_edges)
    pos = nx.bfs_layout(g, start=start_node)

    nx.draw_networkx_nodes(g, pos, node_size=node_size, nodelist=hypergraph["non_terminal_nodes"],
                           ax=axs, node_color=non_terminal_nodes_color, node_shape='s')

    nx.draw_networkx_nodes(g, pos, node_size=node_size, nodelist=hypergraph["terminal_nodes"],
                           ax=axs, node_color=terminal_nodes_color, node_shape='o')

    nx.draw_networkx_edges(g, pos, ax=axs, edge_color=choice_edges_color, edgelist=hypergraph["choice_edges"],
                           arrowstyle=arrow_style, arrowsize=arrows_size, width=arrow_width, style=choice_edges_style,
                           min_target_margin=min_target_margin)

    nx.draw_networkx_edges(g, pos, ax=axs, edgelist=hypergraph["arg_edges"], arrowstyle=arrow_style,
                           arrowsize=arrows_size, width=arrow_width, min_target_margin=min_target_margin,
                           style=arg_edges_style, edge_color=arg_edges_color)

    labels = {node: str(node) for node in all_nodes}
    nx.draw_networkx_labels(g, pos, labels,
                            ax=axs, font_color='white', font_size=node_font_size)

    legend_elements = [
        Line2D([0], [0], color=choice_edges_color, lw=2, linestyle=choice_edges_style, label='Choice'),
        Line2D([0], [0], color=arg_edges_color, lw=2, label='Arg', linestyle=arg_edges_style),
        Patch(facecolor=non_terminal_nodes_color, edgecolor=non_terminal_nodes_color, label="Non-terminals"),
        Line2D([0], [0], marker='o', color=terminal_nodes_color, label='Terminals', markerfacecolor=terminal_nodes_color, markersize=20),
    ]

    axs.legend(handles=legend_elements, loc=legend_loc)

    plt.tight_layout()
    plt.savefig(out_name, dpi=out_dpi)
    plt.show()
