import logging
from typing import Type, Dict, Any, Tuple

from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.mcts.game import OnePlayerGame
from cls_luigi.search.mcts.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy


class NodeFactory:
    def __init__(self, game):
        self.game = game

    def create_node(
        self,
        name: Tuple[str],
        params: Dict[str, Any],
        node_factory: Type['NodeFactory'],
        selection_policy_cls: Type[SelectionPolicy],
        expansion_policy_cls: Type[ExpansionPolicy],
        simulation_policy_cls: Type[SimulationPolicy] = None,
        parent: Type['Node'] = None,
        action_taken: str = None,

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
        simulation_policy_cls: Type[SimulationPolicy]=None,
        node_id: int = None,
        parent: Type[NodeBase] = None,
        action_taken: str = None,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:

        super().__init__(name, logger, **kwargs)

        self.sim_path = None
        self.game = game
        self.params = params
        self.parent = parent
        self.action_taken = action_taken
        self.is_terminal_term = self.game.is_terminal_term(self.name)

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

    def _set_expandable_actions(self):
        self.expandable_actions = self.game.get_valid_actions(self.name)

    def __repr__(self):
        return f"Node: {self.name}"

    def is_fully_expanded(
        self,
        progressiv_widening: float = 0.7
    ) -> bool:

        return len(self.expandable_actions) == 0 and len(self.children) > 0

    def select(
        self
    ) -> Type[NodeBase]:
        best_child = None
        best_score = float("-inf")

        for child in self.children:
            score = self.selection_policy.get_score(child)
            if score > best_score:
                best_child = child
                best_score = score

        return best_child

    def select_best(self):
        best_child = None
        best_score = float("-inf")

        for child in self.children:
            score = child.reward / child.visits
            if score > best_score:
                best_child = child
                best_score = score

        return best_child

    def expand(
        self
    ) -> 'Node':
        self.logger.debug(f"========= expanding: {self.name}")
        sampled_action = self.expansion_policy.get_action()
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
            node_factory=self.node_factory)
        self.children.append(child)
        return child

    def simulate(self, path) -> float:
        sum_rewards = 0
        self.logger.debug(f"========= simulating {self.params['num_simulations']} times: {self.name}")

        for _ in range(self.params["num_simulations"]):
            self.sim_path = path.copy()
            self.logger.debug(f"========= simulation {_}")
            self._simulate(rollout_state=self)
            sum_rewards += self.game.get_reward(self.sim_path)
            self.logger.debug(f"========= simulated path: {self.sim_path}")
            self.sim_path = None

        return sum_rewards / self.params["num_simulations"]

    def _simulate(
        self,
        rollout_state,
    ) -> None:

        if not self.game.is_final_state(rollout_state.name):
            action = self.simulation_policy.get_action(state=rollout_state.name)
            action_node = self.node_factory.create_node(
                params=self.params,
                name=action,
                parent=rollout_state,
                action_taken=action,
                expansion_policy_cls=self.expansion_policy_cls,
                simulation_policy_cls=self.simulation_policy_cls,
                selection_policy_cls=self.selection_policy_cls,
                node_factory=self.node_factory)
            self.sim_path.append(action_node)

            self._simulate(rollout_state=action_node)

    def backprop(
        self,
        reward: float
    ) -> None:
        self.logger.debug(f"========= backpropagating: {self.name}")

        self.visits += 1
        self.reward += reward

        if self.parent:
            self.parent.backprop(reward)
