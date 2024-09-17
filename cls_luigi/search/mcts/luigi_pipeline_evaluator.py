from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Optional, Literal

if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node
    from luigi.task import Task as LuigiTask
    from cls_luigi.tools.constants import MLMAXIMIZATIONMETRICS, MLMINIMIZATIONMETRICS

import logging
import luigi
from luigi.task import flatten
from luigi import build

from cls_luigi.search.core.evaluator import Evaluator


class LuigiPipelineEvaluator(Evaluator):
    def __init__(
            self,
            pipelines: List[LuigiTask],
            metric: Literal[MLMAXIMIZATIONMETRICS.metrics, MLMINIMIZATIONMETRICS.metrics],
            punishment_value: int | float,
            pipeline_timeout: Optional[int] = None,
            logger: Optional[logging.Logger] = None
    ) -> None:
        super().__init__(metric, punishment_value, pipeline_timeout, logger)

        self.pipelines = pipelines
        self._pipeline_map = {}
        self._populate_pipeline_map()
        self._temp_pipeline_key = None
        self.not_found_paths = []
        if self.pipeline_timeout:
            self._set_luigi_worker_configs()

    def _populate_pipeline_map(self) -> None:
        for luigi_pipeline in self.pipelines:
            self._temp_pipeline_key = []
            self._build_pipeline_key(luigi_pipeline)
            self._temp_pipeline_key = tuple(map(lambda x: tuple(x), self._temp_pipeline_key))
            self._pipeline_map[self._temp_pipeline_key] = luigi_pipeline
            self._temp_pipeline_key = None

    def _build_pipeline_key(self, task: LuigiTask, level: int = 0) -> None:
        children = tuple(flatten(task.requires()))

        if level == 0:
            self._temp_pipeline_key.append([])

        self._temp_pipeline_key[level].append(task.__class__.__name__)

        if children:
            level += 1
            if len(self._temp_pipeline_key) < level + 1:
                self._temp_pipeline_key.insert(level, [])
            for index, child in enumerate(children):
                self._build_pipeline_key(task=child, level=level)

    def _get_luigi_pipeline(self, path: List[Node]):
        path = list(filter(lambda x: x.is_terminal_term, path))
        path = tuple(map(lambda x: x.name, path))
        return self._pipeline_map.get(path)

    def evaluate(
            self,
            path: List[Node],
            return_test_score: bool = True
    ) -> float | int:

        luigi_pipeline = self._get_luigi_pipeline(path)
        if luigi_pipeline:
            if luigi_pipeline not in self.evaluated:
                self.evaluated.append(luigi_pipeline)
            try:

                build([luigi_pipeline], local_scheduler=True)
                score = luigi_pipeline.get_score(self.metric)
                if return_test_score:
                    return score["test"]

                return score["train"]

            except Exception as e:
                self.failed[tuple(path)] = luigi_pipeline
                self.logger.debug("Pipeline evaluation failed!")
                self.logger.debug(f"Returned Exception: {e}")
                return self.punishment_value

        if path not in self.not_found_paths:
            self.not_found_paths.append(path)
        self.logger.debug("Pipeline doesn't exists!")
        return self.punishment_value

    def _set_luigi_worker_configs(self):
        if self.pipeline_timeout:
            luigi.configuration.get_config().remove_section("worker")
            luigi.configuration.get_config().set('worker', 'timeout', str(self.pipeline_timeout))

    def reset(self):
        self.evaluated = []
        self.failed = {}
        self.not_found_paths = []
        self._temp_pipeline_key = None
        self.pipeline_timeout = None
