import math
import random
from math import sqrt
from typing import Type

from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.core.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy
from cls_luigi.search.mcts.game import OnePlayerGame


class UCB1(SelectionPolicy):
    def __init__(
        self,
        node: NodeBase,
        exploration_param: float,
        **kwargs
    ) -> None:
        super().__init__(node, **kwargs)
        self.exploration_param = exploration_param

    def select(
        self
    ) -> Type[NodeBase]:

        best_child = None
        best_score = float("-inf")

        for child in self.node.children:
            score = self.get_score(child)
            if score > best_score:
                best_child = child
                best_score = score

        return best_child

    def get_score(
        self,
        child: NodeBase
    ) -> float:

        if child.visits == 0:
            print()
        exploitation = child.reward / child.visits
        exploration = sqrt(math.log(child.parent.visits) / child.visits)

        return exploitation + self.exploration_param * exploration


class RandomExpansion(ExpansionPolicy):
    def __init__(
        self,
        node: NodeBase,
        **kwargs
    ) -> None:
        super().__init__(node, **kwargs)

    def get_action(
        self
    ) -> Type[NodeBase]:
        if len(self.node.expandable_actions) == 0:
            return None
        sampled = random.choice(self.node.expandable_actions)
        self.node.expandable_actions.remove(sampled)
        return sampled


class RandomSimulation(SimulationPolicy):
    def __init__(
        self,
        game: OnePlayerGame,
        **kwargs
    ) -> None:
        super().__init__(game, **kwargs)

    def get_action(
        self,
        state: NodeBase
    ) -> Type[NodeBase] | None:
        valid_moves = self.game.get_valid_actions(state.name, state.parent.name)
        if len(valid_moves) == 0:
            return None
        action = random.choice(valid_moves)
        return action
