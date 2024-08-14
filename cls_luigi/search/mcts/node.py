import logging
from typing import Type, Dict, Any, List

from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.core.policy import SelectionPolicy
from cls_luigi.search.mcts.game import OnePlayerGame
from cls_luigi.search.mcts.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy


class NodeFactory:
    def __init__(self, game):
        self.game = game

    def create_node(
        self,
        name: str,
        params: Dict[str, Any],
        node_factory: Type['NodeFactory'],
        selection_policy_cls: Type[SelectionPolicy],
        expansion_policy_cls: Type[ExpansionPolicy],
        simulation_policy_cls: Type[SimulationPolicy],
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
        params: Dict[str, Any], name: str,
        selection_policy_cls: Type[SelectionPolicy],
        expansion_policy_cls: Type[ExpansionPolicy],
        simulation_policy_cls: Type[SimulationPolicy],
        node_id: int = None,
        parent: Type['Node'] = None,
        action_taken: str = None,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:

        super().__init__(logger, **kwargs)

        self.game = game
        self.params = params
        self.name = name
        self.parent = parent
        self.action_taken = action_taken

        self.selection_policy_cls = selection_policy_cls
        self.expansion_policy_cls = expansion_policy_cls
        self.simulation_policy_cls = simulation_policy_cls

        self.selection_policy = self.selection_policy_cls(
            node=self,
            exploration_param=self.params["exploration_param"])

        self.expansion_policy = self.expansion_policy_cls(
            node=self)
        self.simulation_policy = simulation_policy_cls(self.game)

        self.node_id = node_id
        self.expandable_actions = self.game.get_valid_actions(self.name)

        self.node_factory = node_factory

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

        return self.selection_policy.select()

    def expand(
        self
    ) -> 'Node':
        self.logger.debug(f"========= expanding: {self.name}")

        sampled_action = self.expansion_policy.get_action()
        child_state = sampled_action

        child = self.node_factory.create_node(
            params=self.params,
            name=child_state,
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
            self.logger.debug(f"========= simulation {_}")
            sum_rewards += self._simulate(path=path)

        return sum_rewards / self.params["num_simulations"]

    def _simulate(
        self,
        path: List[str]
    ) -> float:

        rollout_state = self
        self.logger.debug(f"========= simulating: {rollout_state.name}")

        while True:
            if self.game.is_final_state(rollout_state.name):
                self.logger.debug(f"========= terminal state: {rollout_state.name}")
                return self.game.get_reward(path)

            action = self.simulation_policy.get_action(state=rollout_state.name)
            path.append(rollout_state.name)
            path.append(action)
            is_final = self.game.is_final_state(action)

            if is_final:
                self.logger.debug(f"========= terminal state: {rollout_state.name} with action: {action}")
                return self.game.get_reward(path)

            self.logger.debug(f"========= non terminal state: {rollout_state.name} with action: {action}")

            rollout_parent = rollout_state
            rollout_state = self.node_factory.create_node(
                params=self.params,
                name=self.simulation_policy.get_action(state=action),
                parent=rollout_parent,
                action_taken=action,
                expansion_policy_cls=self.expansion_policy_cls,
                simulation_policy_cls=self.simulation_policy_cls,
                selection_policy_cls=self.selection_policy_cls,
                node_factory=self.node_factory)

    def backprop(
        self,
        reward: float
    ) -> None:
        self.logger.debug(f"========= backpropagating: {self.name}")

        self.visits += 1
        self.reward += reward

        if self.parent:
            self.parent.backprop(reward)
