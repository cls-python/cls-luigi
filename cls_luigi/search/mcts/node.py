from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Type, Dict, Any, Tuple, List, Optional

if TYPE_CHECKING:
    from cls_luigi.search.mcts.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy
    from cls_luigi.search.mcts.game import OnePlayerGame


import logging
from cls_luigi.search.core.node import NodeBase


class NodeFactory:
    def __init__(self, game):
        self.game = game

    def create_node(
        self,
        name: Tuple[str],
        params: Dict[str, Any],
        node_factory: NodeFactory,
        selection_policy_cls: Type[SelectionPolicy],
        expansion_policy_cls: Type[ExpansionPolicy],
        simulation_policy_cls: Optional[Type[SimulationPolicy]] = None,
        parent: Optional[Node] = None,
        action_taken: Optional[str] = None,
        prog_widening_params: Optional[Dict[str, Any]] = None,

    ):
        return Node(
            name=name,
            game=self.game,
            node_factory=node_factory,
            params=params,
            selection_policy_cls=selection_policy_cls,
            expansion_policy_cls=expansion_policy_cls,
            simulation_policy_cls=simulation_policy_cls,
            parent=parent,
            action_taken=action_taken,
            prog_widening_params=prog_widening_params,
        )


class Node(NodeBase):
    selection_policy: SelectionPolicy

    def __init__(
        self,
        game: Type[OnePlayerGame],
        node_factory: Type[NodeFactory],
        params: Dict[str, Any],
        name: Tuple[str],
        selection_policy_cls: Type[SelectionPolicy],
        expansion_policy_cls: Type[ExpansionPolicy],
        simulation_policy_cls: Optional[Type[SimulationPolicy]] = None,
        node_id: Optional[int] = None,
        parent: Optional[Node] = None,
        action_taken: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        prog_widening_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:

        super().__init__(name, logger, **kwargs)

        self.sim_path = None
        self.game = game
        self.params = params
        self.parent = parent
        self.action_taken = action_taken
        self.is_terminal_term = self.game.is_terminal_term(self)

        self.selection_policy_cls = selection_policy_cls
        self.expansion_policy_cls = expansion_policy_cls
        self.simulation_policy_cls = simulation_policy_cls

        self.selection_policy = self.selection_policy_cls(
            node=self,
            exploration_param=self.params["exploration_param"])

        self.expansion_policy = self.expansion_policy_cls(
            node=self)
        if self.simulation_policy_cls:
            self.simulation_policy = simulation_policy_cls(self.game)

        self.node_id = node_id
        self.expandable_actions = None
        self._set_expandable_actions()
        self.node_factory = node_factory
        self.prog_widening_params = prog_widening_params

    def _set_expandable_actions(self) -> None:
        self.logger.debug(f"Setting expandable actions for node: {self.name}")
        self.expandable_actions = self.game.get_valid_actions(self)

    def __repr__(self) -> str:
        return f"Node: {self.name}"

    def is_fully_expanded(
        self,
    ) -> bool:
        if not self.prog_widening_params:
            return self._is_fully_expanded_default()
        else:
            self.logger.debug(f"progressive widening is activated")
            return self.is_fully_expanded_progressive_widening()

    def _is_fully_expanded_default(self) -> bool:
        """
        default check if the node is fully expanded. activated when no fully_expanded_params are provided.
        """
        return len(self.expandable_actions) == 0 and len(self.children) > 0

    def is_fully_expanded_progressive_widening(self) -> bool:
        """
        This method is used to determine if a node is fully expanded based on the progressive widening strategy.

        The self.fully_expanded_params dictionary should contain the following
        keys: threshold (int), progressiv_widening_coeff (int or float), max_children (int).


        Returns:
            bool: True if the node is fully expanded, False otherwise.
        """
        if self.visits == 0:
            return False

        if not self.expandable_actions:
            return True

        threshold_value = self.prog_widening_params["threshold"] * (
            self.visits ** self.prog_widening_params["progressiv_widening_coeff"])
        if len(self.children) < min(threshold_value, self.prog_widening_params["max_children"]):
            return False
        return True

    def select(
        self
    ) -> Node:
        best_child = None
        best_score = None

        for child in self.children:
            if not best_child:
                best_child = child
                best_score, _ = self.selection_policy.get_score(child)
                continue

            score, explanation = self.selection_policy.get_score(child)
            child.explanations.append(explanation)

            if score > best_score:
                best_child = child
                best_score = score

        return best_child

    def expand(
        self
    ) -> Node:
        sampled_action = self.expansion_policy.get_action()
        self.logger.debug(f"Sampled child action {sampled_action} for node {self.name}")

        if isinstance(sampled_action, str):
            sampled_action = (sampled_action,)

        child = self.node_factory.create_node(
            params=self.params,
            name=sampled_action,
            parent=self,
            action_taken=sampled_action,
            expansion_policy_cls=self.expansion_policy_cls,
            simulation_policy_cls=self.simulation_policy_cls,
            selection_policy_cls=self.selection_policy_cls,
            node_factory=self.node_factory,
            prog_widening_params=self.prog_widening_params
        )
        self.children.append(child)
        self.logger.debug(f"Created child action {child.name} for node {self.name}")
        return child

    def simulate(self, path: List[Node]) -> float:
        sum_rewards = 0
        self.logger.debug(f"Simulating{self.params['num_simulations']} times: {self.name}")

        for ix_sim in range(self.params["num_simulations"]):
            self.sim_path = path.copy()
            self.logger.debug(f"Simulation {ix_sim}")
            self._simulate(rollout_state=self)
            sum_rewards += self.game.evaluate(self.sim_path)
            self.logger.debug(f"Simulated path: {self.sim_path}")
            self.sim_path = None

        return sum_rewards / self.params["num_simulations"]

    def _simulate(
        self,
        rollout_state: Node,
    ) -> None:

        if not self.game.is_final_state(rollout_state):
            action = self.simulation_policy.get_action(state=rollout_state)
            action_node = self.node_factory.create_node(
                params=self.params,
                name=action,
                parent=rollout_state,
                action_taken=action,
                expansion_policy_cls=self.expansion_policy_cls,
                simulation_policy_cls=self.simulation_policy_cls,
                selection_policy_cls=self.selection_policy_cls,
                node_factory=self.node_factory,
                prog_widening_params=self.prog_widening_params)
            self.sim_path.append(action_node)

            self._simulate(rollout_state=action_node)

    def backprop(
        self,
        reward: float
    ) -> None:
        self.logger.debug(f"Backpropagating: {self.name} with reward {reward}")

        self.visits += 1
        self.sum_rewards += reward

        if self.parent:
            self.parent.backprop(reward)
