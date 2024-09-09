from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from typing import Any

if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node
    from cls_luigi.search.mcts.game import OnePlayerGame

from cls_luigi.search.core.policy import SelectionPolicy, ExpansionPolicy, SimulationPolicy

import logging
import math
import random
from math import sqrt


class UCT(SelectionPolicy):
    def __init__(
        self,
        node: Node,
        exploration_param: float,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:
        super().__init__(node, logger, **kwargs)
        self.exploration_param = exploration_param

    def get_score(
        self,
        child: Node
    ) -> tuple[float | Any, dict[str, float | None | Any]]:
        exploitation = child.sum_rewards / child.visits
        exploration = sqrt(math.log(child.parent.visits) / child.visits)
        # exploration = sqrt(child.parent.visits / child.visits)
        score = exploitation + self.exploration_param * exploration
        if self._regularization() is not None:
            score *= self._regularization()

        explanation = {
            "exploitation": exploitation,
            "exploration": exploration,
            "exploration_param": self.exploration_param,
            "score": score,
        }

        return score, explanation


class RandomExpansion(ExpansionPolicy):
    def __init__(
        self,
        node: Node,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:
        super().__init__(node, logger, **kwargs)

    def get_action(
        self
    ) -> Optional[Node]:
        if len(self.node.expandable_actions) == 0:
            return None
        sampled = random.choice(self.node.expandable_actions)
        self.node.expandable_actions.remove(sampled)
        self.logger.debug(f"========= sampled action: {sampled}")

        return sampled


class RandomSimulation(SimulationPolicy):
    def __init__(
        self,
        game: OnePlayerGame,
        logger: logging.Logger = None,
        **kwargs
    ) -> None:
        super().__init__(game, logger, **kwargs)

    def get_action(
        self,
        state: Node
    ) -> Optional[Node]:
        valid_actions = self.game.get_valid_actions(state)
        if len(valid_actions) == 0:
            self.logger.debug(f"========= no valid actions for: {state}")

            return None
        action = random.choice(valid_actions)
        self.logger.debug(f"========= simulated action: {action}")

        return action
