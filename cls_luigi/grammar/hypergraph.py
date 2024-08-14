from typing import Dict, List, Tuple

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def get_hypergraph_dict_from_tree_grammar(
    tree_grammar: Dict[str, str | List[str] | Dict[str, List[str]]]
) -> Dict[str, List[str]]:
    hypergraph = {
        "start": tree_grammar["start"],
        "choice_edges": [],
        "arg_edges": [],
        "non_terminal_nodes": tree_grammar["non_terminals"],
        "terminal_nodes": tree_grammar["terminals"]}

    for lhs, rhs in tree_grammar["rules"].items():
        for combinator, args in rhs.items():
            hypergraph["choice_edges"].append((lhs, combinator))
            for arg in args:
                hypergraph["arg_edges"].append((combinator, arg))

    return hypergraph


def build_hypergraph(hyper_graph_dict: Dict[str, List[str]]):
    g = nx.DiGraph()
    g.add_nodes_from(hyper_graph_dict["non_terminal_nodes"], terminal_node=False, start_node=False)
    g.add_nodes_from(hyper_graph_dict["terminal_nodes"], terminal_node=True, start_node=False)

    g.add_edges_from(hyper_graph_dict["arg_edges"], arg_edge=True)
    g.add_edges_from(hyper_graph_dict["choice_edges"], arg_edge=False)

    g.nodes[hyper_graph_dict["start"]]["start_node"]= True

    return g


def plot_hypergraph_components(
    # hypergraph: Dict[str, List[str]],
    g: nx.DiGraph,
    out_name: str,
    start_node: str,
    node_font_size: int = 5,
    node_size: int = 5000,
    figsize: Tuple[int, int] = (13, 8),
    facecolor: str = "White",
    non_terminal_nodes_color: str = "#003049",
    terminal_nodes_color: str = "#b87012",
    start_node_color: str = "#556B2F",
    choice_edges_color: str = "#818589",
    arg_edges_color: str = "black",
    choice_edges_style: str = "dotted",
    arg_edges_style: str = "solid",
    arrow_style: str = "->",
    arrows_size: int = 28,
    arrow_width: int = 2,
    min_target_margin: int = 38,
    legend_loc: str = 'best',
    out_dpi: int = 600,
    plot_title: str = "Tree Grammar as Hypergraph",
    non_terminal_node_shape: str = 's',
    terminal_node_shape: str = 'o',
    node_label_color="white",
    title_fontsize: int = 15,
    title_font_weight: str = 'bold',
    title_loc: str = 'center',

) -> None:
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.axis('off')
    fig.patch.set_facecolor(facecolor)

    terminal_nodes = [node for node in g.nodes if g.nodes[node]["terminal_node"]]
    non_terminal_nodes = [node for node in g.nodes if not g.nodes[node]["terminal_node"]]
    start_node = [node for node in g.nodes if g.nodes[node]["start_node"]]

    choice_edges = [edge for edge in g.edges if not g.edges[edge]["arg_edge"]]
    arg_edges = [edge for edge in g.edges if g.edges[edge]["arg_edge"]]

    start_node = [node for node in g.nodes if g.nodes[node]["start_node"]]

    pos = nx.bfs_layout(g, start=start_node)
    #
    nx.draw_networkx_nodes(g, pos, node_size=node_size, nodelist=non_terminal_nodes,
                           ax=axs, node_color=non_terminal_nodes_color, node_shape=non_terminal_node_shape)

    nx.draw_networkx_nodes(g, pos, node_size=node_size, nodelist=terminal_nodes,
                           ax=axs, node_color=terminal_nodes_color, node_shape=terminal_node_shape)

    nx.draw_networkx_nodes(g, pos, node_size=node_size, nodelist=start_node,
                           ax=axs, node_color=start_node_color, node_shape=non_terminal_node_shape)

    nx.draw_networkx_edges(g, pos, ax=axs, edge_color=choice_edges_color, edgelist=choice_edges,
                           arrowstyle=arrow_style, arrowsize=arrows_size, width=arrow_width, style=choice_edges_style,
                           min_target_margin=min_target_margin)

    nx.draw_networkx_edges(g, pos, ax=axs, edgelist=arg_edges, arrowstyle=arrow_style,
                           arrowsize=arrows_size, width=arrow_width, min_target_margin=min_target_margin,
                           style=arg_edges_style, edge_color=arg_edges_color)

    labels = {node: str(node) for node in g.nodes}
    nx.draw_networkx_labels(g, pos, labels,
                            ax=axs, font_color=node_label_color, font_size=node_font_size)

    legend_elements = [
        Line2D([0], [0], color=choice_edges_color, lw=2, linestyle=choice_edges_style, label='Choice'),
        Line2D([0], [0], color=arg_edges_color, lw=2, label='Argument', linestyle=arg_edges_style),
        Patch(facecolor=non_terminal_nodes_color, edgecolor=non_terminal_nodes_color, label="Non-terminal"),
        Line2D([0], [0], marker=terminal_node_shape, color=terminal_nodes_color, label='Terminal',
               markerfacecolor=terminal_nodes_color, markersize=20, linewidth=0),
        Patch(facecolor=start_node_color, edgecolor=start_node_color, label="Start"),

    ]
    axs.set_title(plot_title, fontsize=title_fontsize, fontweight=title_font_weight, loc=title_loc)
    axs.legend(handles=legend_elements, loc=legend_loc)

    plt.tight_layout()
    plt.savefig(out_name, dpi=out_dpi)
    plt.show()
