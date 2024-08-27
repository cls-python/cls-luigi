import abc
import logging
import pickle
from typing import Dict, List, Type, Any

from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.core.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy
from cls_luigi.search.core.tree import TreeBase
from cls_luigi.search.mcts.game import OnePlayerGame
from cls_luigi.search.mcts.tree import MCTSTreeWithGrammar
from cls_luigi.search.mcts.node import NodeFactory
from cls_luigi.search.mcts.policy import UCT, RandomExpansion


class SinglePlayerMCTS(abc.ABC):
    """Base class for Monte Carlo Tree Search"""

    def __init__(
        self,
        parameters: Dict[str, Any],
        game: Type[OnePlayerGame],
        selection_policy: Type[SelectionPolicy] = UCT,
        expansion_policy: Type[ExpansionPolicy] = RandomExpansion,

        tree_cls: Type[TreeBase] = MCTSTreeWithGrammar,
        node_factory_cls: NodeFactory = NodeFactory,
        simulation_policy: Type[SimulationPolicy] = None,

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
    ) -> List[NodeBase]:
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
