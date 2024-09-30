from __future__ import annotations
from typing import TYPE_CHECKING, Union, Tuple, Type
from typing import List, Optional, Literal

if TYPE_CHECKING:
    from cls_luigi.search.mcts.node import Node
    from luigi.task import Task as LuigiTask
    from cls_luigi.tools.constants import MLMAXIMIZATIONMETRICS, MLMINIMIZATIONMETRICS

from pynisher import TimeoutException
from cls_luigi.tools.luigi_daemon import LinuxLuigiDaemonHandler
import logging
import luigi
from luigi.task import flatten

from cls_luigi.search.core.evaluator import Evaluator

from cls_luigi.tools.constants import SUCCESS, FAILED, TIMEOUT, NOTFOUND, NOSCORE
import pynisher


class LuigiPipelineEvaluator(Evaluator):
    def __init__(
        self,
        tasks: List[LuigiTask],
        metric: Literal[MLMAXIMIZATIONMETRICS.metrics, MLMINIMIZATIONMETRICS.metrics],
        punishment_value: int | float,
        task_timeout: Optional[int] = None,
        pipeline_timeout: Optional[int] = None,
        debugging_mode: bool = False,
        luigi_daemon_handler_cls: Type[LinuxLuigiDaemonHandler] = LinuxLuigiDaemonHandler,
        logger: Optional[logging.Logger] = None
    ) -> None:
        super().__init__(
            metric=metric,
            punishment_value=punishment_value,
            pipeline_timeout=pipeline_timeout,
            task_timeout=task_timeout,
            debugging_mode=debugging_mode,
            logger=logger
        )

        self.tasks = tasks
        self._pipeline_map = {}
        self.populate_pipeline_map()
        self._temp_task_key = None
        if self.component_timeout:
            self._set_luigi_worker_configs()
        self.luigi_daemon_handler = luigi_daemon_handler_cls(logger=self.logger)

        self.use_local_scheduler = True

        if not self.debugging_mode:
            self.luigi_daemon_handler.start_luigi_server()
            self.use_local_scheduler = False


    def populate_pipeline_map(self) -> None:
        for task in self.tasks:
            self._temp_task_key = []
            self._build_pipeline_key(task)
            self._temp_task_key = tuple(map(lambda x: tuple(x), self._temp_task_key))
            self._pipeline_map[self._temp_task_key] = task
            self._temp_task_key = None

    def _build_pipeline_key(self, task: LuigiTask, level: int = 0) -> None:
        children = tuple(flatten(task.requires()))

        if level == 0:
            self._temp_task_key.append([])

        self._temp_task_key[level].append(task.__class__.__name__)

        if children:
            level += 1
            if len(self._temp_task_key) < level + 1:
                self._temp_task_key.insert(level, [])
            for index, child in enumerate(children):
                self._build_pipeline_key(task=child, level=level)

    def _get_luigi_task(self, path: List[Node]):
        path = list(filter(lambda x: x.is_terminal_term, path))
        path = tuple(map(lambda x: x.name, path))
        return self._pipeline_map.get(path)

    def _handle_timeout(
        self,
        path: List[Node],
        task: luigi.Task,
    ) -> Tuple[str, str, Union[int, float]]:
        path = tuple(path)
        if path not in self.timed_out:
            self.timed_out[path] = task

        return task.task_id, TIMEOUT, self.punishment_value

    def _handle_failed(
        self,
        path: List[Node],
        task: luigi.Task
    ) -> Tuple[str, str, Union[int, float]]:
        path = tuple(path)
        if path not in self.failed:
            self.failed[path] = task
        return task.task_id, FAILED, self.punishment_value

    def _handle_not_found_pipelines(
        self,
        path: List[Node],
    ) -> Tuple[Optional[str], str, Union[int, float]]:
        if path not in self.not_found:
            self.not_found.append(path)
        return None, NOTFOUND, self.punishment_value

    def _schedule_and_run_pipeline(
        self,
        task: luigi.Task,
        detailed_summary: bool = False
    ) -> None:
        luigi.build([task], local_scheduler=self.use_local_scheduler, detailed_summary=detailed_summary)

    def evaluate(
        self,
        path: List[Node],
        return_test_score: bool = True
    ) -> Tuple[Optional[str], str, Union[int, float]]:

        task = self._get_luigi_task(path)
        if task:
            try:
                with pynisher.limit(self._schedule_and_run_pipeline, wall_time=self.pipeline_timeout) as running_task:
                    running_task(task)
                    scores = task.get_score(self.metric)

            except TimeoutException as e:
                self.logger.debug(f"Pipeline Evaluation timed out:\n{e}")
                return self._handle_timeout(path, task)

            except (FileNotFoundError, Exception) as e:
                self.logger.debug(f"Pipeline Evaluation Failed with the Exception:\n{e}")
                return self._handle_failed(path, task)

            if task not in self.evaluated:
                self.evaluated[tuple(path)] = task

            task_id = task.task_id
            status = SUCCESS
            score = scores["test"]
            if not return_test_score:
                score = scores["train"]

            return task_id, status, score

        else:
            self.logger.debug("Pipeline doesn't exists!")
            return self._handle_not_found_pipelines(path)

    def _set_luigi_worker_configs(self):
        if self.component_timeout:
            luigi.configuration.get_config().remove_section("worker")
            luigi.configuration.get_config().set('worker', 'timeout', str(self.component_timeout))

    def reset(self):
        self.evaluated = {}
        self.failed = {}
        self.timed_out = {}
        self.not_found = []
        self._temp_task_key = None
        self.component_timeout = None

    def add_evaluated_failed_timed_out(self, summary_dict):
        summary_dict["evaluated"] = {
            str(key): value.task_id
            for key, value in self.evaluated.items()
        }

        summary_dict["failed"] = {
            str(key): value.task_id
            for key, value in self.failed.items()
        }

        summary_dict["timed_out"] = {
            str(key): value.task_id
            for key, value in self.timed_out.items()
        }

    def get_json_ready_summary(self):
        summary_dict = {
            "metrics": self.metric,
            "punishment_value": self.punishment_value,
            "pipeline_map": {},  #
            "evaluated": {},
            "failed": {},
            "timed_out": {},
            "not_found": [node.name for pipeline in self.not_found for node in pipeline]
        }
        for key, value in self._pipeline_map.items():
            summary_dict["pipeline_map"][str(key)] = value.task_id

        self.add_evaluated_failed_timed_out(summary_dict)
        return summary_dict

    def __del__(self):
        self.luigi_daemon_handler.shutdown_luigi_server()
