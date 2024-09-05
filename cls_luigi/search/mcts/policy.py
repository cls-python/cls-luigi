import logging
import math
import random
from math import sqrt
from typing import Type

from cls_luigi.search.core.node import NodeBase
from cls_luigi.search.core.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy
from cls_luigi.search.mcts.game import OnePlayerGame


class UCT(SelectionPolicy):
    def __init__(
        self,
        node: NodeBase,
        exploration_param: float,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:
        super().__init__(node, logger, **kwargs)
        self.exploration_param = exploration_param

    def get_score(
        self,
        child: NodeBase
    ) -> float:
        exploitation = child.sum_rewards / child.visits
        # exploration = sqrt(math.log(child.parent.visits) / child.visits)
        exploration = sqrt(child.parent.visits / child.visits)
        score = exploitation + self.exploration_param * exploration

        explanation = {
            "exploitation": exploitation,
            "exploration": exploration,
            "exploration_param": self.exploration_param,
            "score": score
        }

        return score, explanation


class RandomExpansion(ExpansionPolicy):
    def __init__(
        self,
        node: NodeBase,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:
        super().__init__(node, logger, **kwargs)

    def get_action(
        self
    ) -> Type[NodeBase] | None:
        if len(self.node.expandable_actions) == 0:
            return None
        sampled = random.choice(self.node.expandable_actions)
        self.node.expandable_actions.remove(sampled)
        self.logger.debug(f"========= sampled action: {sampled}")

        return sampled


class RandomSimulation(SimulationPolicy):
    def __init__(
        self,
        game: Type[OnePlayerGame],
        logger: logging.Logger = None,
        **kwargs
    ) -> None:
        super().__init__(game, logger, **kwargs)

    def get_action(
        self,
        state: NodeBase
    ) -> Type[NodeBase] | None:
        valid_moves = self.game.get_valid_actions(state)
        if len(valid_moves) == 0:
            self.logger.debug(f"========= no valid actions for: {state}")

            return None
        action = random.choice(valid_moves)
        self.logger.debug(f"========= simulated action: {action}")

        return action
