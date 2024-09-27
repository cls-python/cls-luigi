from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from typing import Dict, List, Type, Any

import pandas as pd

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
        prog_widening_params: Optional[Dict[str, Any]] = None,
        out_path: Optional[str] = None,

        logger: logging.Logger = None
    ) -> None:

        self.game = game
        self.node_factory_cls = node_factory_cls
        self.node_factory = self.node_factory_cls(self.game)
        self.parameters = parameters
        self.selection_policy = selection_policy
        self.expansion_policy = expansion_policy
        self.simulation_policy = simulation_policy
        self.prog_widening_params = prog_widening_params
        self.tree = tree_cls(root=self.get_root_node(), hypergraph=self.game.hypergraph)
        self.out_path = out_path
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.incumbent = (None, None, None)
        self.run_history = pd.DataFrame(columns=["iteration", "luigi_id", "mcts_path", "status", "score"])
        # self.mcts_scenario = {
        #     "type": self.__class__.__name__,
        #     "parameters": self.parameters,
        #     "policies": {
        #         "selection": self.selection_policy.__name__,
        #         "expansion": self.expansion_policy.__name__,
        #         "simulation": self.simulation_policy.__name__
        #     },
        #     "component_timeout": 123,
        #     "pipeline_timeout": "accuracy",
        #     "punishment_value": 0.0,
        #
        #     # "pipeline_metric": None
        #     # "filters:": [],
        #     # "sense": "maximize"
        # }

        self.iter_counter = 0


        self.logger.debug(f"Initialized {self.__class__.__name__} with parameters: {self.parameters}")

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
            prog_widening_params=self.prog_widening_params
        )

    def run(
        self
    ) -> List[Node]:
        ...



    def _update_incumbent(
        self,
        path: List[Node],
        task_id: str,
        reward
    ) -> None:
        curr_path, curr_task_id, curr_reward = self.incumbent

        if (curr_path is None) and (reward != float("inf") or reward != float("-inf")):
            self.logger.debug(f"Setting incumbent for the first time")
            self.incumbent = (path, task_id, reward)
        else:
            if reward > curr_reward:
                self.logger.debug(f"Updating incumbent {path} with higher reward: {reward}")
                self.incumbent = (path, task_id, reward)

            elif reward == curr_reward:
                if len(path) < len(curr_path):
                    self.logger.debug(f"Updating incumbent {path} with same reward: {reward} but shorter path")
                    self.incumbent = (path, task_id, reward)
            else:
                self.logger.debug(
                    f"Current incumbent {curr_path} with reward: {curr_reward} remains unchanged")

    def draw_tree(
        self,
        out_path: Optional[str] = None,
        plot: bool = False,
        *args
    ) -> None:

        best_path = self.incumbent[0]
        self.tree.render(out_name=out_path, plot=plot, node_size=1500, best_path=best_path, *args)

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
