from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from typing import Dict, List, Type, Any

if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node
    from cls_luigi.search.core.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy
    from cls_luigi.search.core.tree import TreeBase
    from cls_luigi.search.mcts.game import OnePlayerGame

    from cls_luigi.search.mcts.game import OnePlayerGame

from cls_luigi.search.mcts.tree import MCTSTreeWithGrammar
from cls_luigi.search.mcts.node import NodeFactory
from cls_luigi.search.mcts.policy import UCT, RandomExpansion

import abc
import logging
import pickle



class SinglePlayerMCTS(abc.ABC):
    """Base class for Monte Carlo Tree Search"""

    def __init__(
        self,
        parameters: Dict[str, Any],
        game: OnePlayerGame,
        selection_policy: Type[SelectionPolicy] = UCT,
        expansion_policy: Type[ExpansionPolicy] = RandomExpansion,
        tree_cls: Type[TreeBase] = MCTSTreeWithGrammar,
        node_factory_cls: Type[NodeFactory] = NodeFactory,
        simulation_policy: Optional[Type[SimulationPolicy]] = None,
        fully_expanded_params: Optional[Dict[str, Any]] = None,

        logger: logging.Logger = None
    ) -> None:

        self.game = game
        self.node_factory_cls = node_factory_cls
        self.node_factory = self.node_factory_cls(self.game)
        self.parameters = parameters
        self.selection_policy = selection_policy
        self.expansion_policy = expansion_policy
        self.simulation_policy = simulation_policy
        self.fully_expanded_params = fully_expanded_params
        self.tree = tree_cls(root=self.get_root_node(), hypergraph=self.game.hypergraph)
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.current_incumbent_and_score = None

    def get_root_node(
        self
    ) -> Node:

        return self.node_factory.create_node(
            params=self.parameters,
            name=self.game.get_initial_state(),
            selection_policy_cls=self.selection_policy,
            expansion_policy_cls=self.expansion_policy,
            simulation_policy_cls=self.simulation_policy,
            node_factory=self.node_factory,
            fully_expanded_params=self.fully_expanded_params
        )

    def run(
        self
    ) -> List[Node]:
        ...

    def get_incumbent(
        self
    ) -> List[Node]:
        if self.current_incumbent_and_score is not None:
            return self.current_incumbent_and_score[0]
        return []

    def _update_incumbent(
        self,
        path: List[Node],
        reward
    ) -> None:
        if (self.current_incumbent_and_score is None) and (reward != float("inf") or reward != float("-inf")):
            self.logger.debug(f"Setting incumbent for the first time")
            self.current_incumbent_and_score = (path, reward)
        else:
            if reward > self.current_incumbent_and_score[1]:
                self.logger.debug(f"Updating incumbent {path} with higher reward: {reward}")
                self.current_incumbent_and_score = (path, reward)

            elif reward == self.current_incumbent_and_score[1]:
                if len(path) < len(self.current_incumbent_and_score[0]):
                    self.logger.debug(f"Updating incumbent {path} with same reward: {reward} but shorter path")
                    self.current_incumbent_and_score = (path, reward)
            else:
                self.logger.debug(
                    f"Current incumbent {self.current_incumbent_and_score[0]} with reward: {self.current_incumbent_and_score[1]} remains unchanged")

    def draw_tree(
        self,
        out_path: Optional[str] = None,
        plot: bool = False,
        *args) -> None:
        best_path = self.get_incumbent()
        self.tree.draw_tree(out_name=out_path, plot=plot, node_size=1500, best_path=best_path, *args)

    def shut_down(
        self,
        mcts_path: Optional[str] = None,
        tree_path: Optional[str] = None
    ) -> None:

        self.logger.debug("Shutting down SP-MCTS")
        if mcts_path:
            with open(mcts_path, "wb") as f:
                pickle.dump(self, f)
        self.logger.debug(f"Saved MCTS object as pickle file to {mcts_path}")
        if tree_path:
            self.tree.save(tree_path)
