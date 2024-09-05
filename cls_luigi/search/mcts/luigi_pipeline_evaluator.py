import logging
import pickle
from typing import List
import luigi
from luigi.task import flatten

from cls_luigi.search.core.evaluator import Evaluator


class LuigiPipelineEvaluator(Evaluator):
    def __init__(
        self,
        pipelines: List[luigi.Task],
        punishment_value: int | float,
        logger: logging.Logger = None
    ) -> None:
        super().__init__(punishment_value, logger)

        self.pipelines = pipelines
        self._pipeline_map = {}
        self._populate_pipeline_map()
        self._temp_pipeline_key = None
        self.not_found_paths = []

    def _populate_pipeline_map(self) -> None:
        for luigi_pipeline in self.pipelines:
            self._temp_pipeline_key = []
            self._build_pipeline_key(luigi_pipeline)
            self._temp_pipeline_key = tuple(map(lambda x: tuple(x), self._temp_pipeline_key))
            self._pipeline_map[self._temp_pipeline_key] = luigi_pipeline
            self._temp_pipeline_key = None

    def _build_pipeline_key(self, task: luigi.task, level: int = 0) -> None:
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

    def _get_luigi_pipeline(self, path):
        path = list(filter(lambda x: x.is_terminal_term, path))
        path = tuple(map(lambda x: x.name, path))
        return self._pipeline_map.get(path)

    def evaluate(
        self,
        path: List["NodeBase"]
    ) -> float | int:

        luigi_pipeline = self._get_luigi_pipeline(path)
        if luigi_pipeline:
            if luigi_pipeline not in self.evaluated:
                self.evaluated.append(luigi_pipeline)
            try:
                score_pkl_path = luigi_pipeline.output()["score"].path

                luigi.build([luigi_pipeline], local_scheduler=True)

                with open(score_pkl_path, "rb") as in_file:
                    return pickle.load(in_file)

            except FileNotFoundError:
                self.failed[path] = luigi_pipeline
                self.logger.debug("Pipeline evaluation failed!")
                return self.punishment_value

        if path not in self.not_found_paths:
            self.not_found_paths.append(path)
        self.logger.debug("Pipeline doesn't exists!")
        return self.punishment_value
