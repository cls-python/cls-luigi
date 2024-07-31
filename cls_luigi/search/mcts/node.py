import logging
from typing import Type, Dict, Any, List

from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.mcts.game import OnePlayerGame
from cls_luigi.search.mcts.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy


class Node(NodeBase):

    def __init__(
        self,
        game: OnePlayerGame,
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

        super().__init__(**kwargs)

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

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
        self.children = []
        self.visits = 0
        self.reward = 0
        if self.parent:
            self.expandable_actions = self.game.get_valid_actions(self.name, self.parent.name).copy()
        else:
            self.expandable_actions = self.game.get_valid_actions(self.name).copy()

    def is_fully_expanded(
        self
    ) -> bool:

        return len(self.expandable_actions) == 0 and len(self.children) > 0

    def select(
        self
    ) -> 'Node':

        return self.selection_policy.select()

    def __repr__(self):
        return f"{self.name}"

    def expand(
        self
    ) -> 'Node':
        self.logger.debug(f"========= expanding: {self.name}")

        sampled_action = self.expansion_policy.get_action()
        child_state = self.game.get_next_state(sampled_action)
        child = Node(self.game,
                     self.params,
                     child_state,
                     parent=self,
                     action_taken=sampled_action,
                     expansion_policy_cls=self.expansion_policy_cls,
                     simulation_policy_cls=self.simulation_policy_cls,
                     selection_policy_cls=self.selection_policy_cls)
        self.children.append(child)
        return child

    def simulate(
        self,
        path: List['Node']
    ) -> float:
        rollout_state = self
        self.logger.debug(f"========= simulating: {rollout_state.name}")

        while True:
            action = self.simulation_policy.get_action(state=rollout_state)
            path.append(rollout_state)
            path.append(action)
            is_terminal = self.game.is_terminal(rollout_state.parent.name, rollout_state.name, action)

            if is_terminal:
                self.logger.debug(f"========= terminal state: {rollout_state.name} with action: {action}")
                return self.game.get_reward(path)

            self.logger.debug(f"========= non terminal state: {rollout_state.name} with action: {action}")

            rollout_parent = rollout_state
            rollout_state = Node(
                game=self.game,
                params=self.params,
                name=self.game.get_next_state(action),
                selection_policy_cls=self.selection_policy_cls,
                parent=rollout_parent,
                action_taken=action,
                expansion_policy_cls=self.expansion_policy_cls,
                simulation_policy_cls=self.simulation_policy_cls,
            )

    def backprop(
        self,
        reward: float
    ) -> None:
        self.logger.debug(f"========= backpropagating: {self.name}")

        self.visits += 1
        self.reward += reward

        if self.parent:
            self.parent.backprop(reward)
