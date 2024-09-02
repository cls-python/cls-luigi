import logging
import pickle
from typing import List
import luigi
from luigi.task import flatten


class Evaluator:
    def __init__(
        self,
        pipelines: List[luigi.Task],
        logger: logging.logger = None
    ) -> None:
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.pipelines = pipelines
        self.pipeline_map = {}
        self.populate_pipeline_map()
        self.temp_pipeline_key = None


    def populate_pipeline_map(self) -> None:
        for luigi_pipeline in self.pipelines:
            self.temp_pipeline_key = []
            self._build_pipeline_key(luigi_pipeline)
            self.temp_pipeline_key = tuple(map(lambda x: tuple(x), self.temp_pipeline_key))
            self.pipeline_map[self.temp_pipeline_key] = luigi_pipeline
            self.temp_pipeline_key = None

    def _build_pipeline_key(self, task: luigi.task, level: int = 0) -> None:
        print(f"Level: {level}")
        children = tuple(flatten(task.requires()))

        if level == 0:
            self.temp_pipeline_key.append([])

        self.temp_pipeline_key[level].append(task.__class__.__name__)

        if children:
            level += 1
            if len(self.temp_pipeline_key) < level + 1:
                self.temp_pipeline_key.insert(level, [])
            for index, child in enumerate(children):
                self._build_pipeline_key(task=child, level=level)

    def get_luigi_pipeline(self, path):
        path = list(filter(lambda x: x.is_terminal_term, path))
        path = tuple(map(lambda x: x.name, path))
        return self.pipeline_map.get(path)

    def evaluate(
        self,
        path
    ) -> float:

        luigi_pipeline = self.get_luigi_pipeline(path)
        if luigi_pipeline:
            try:
                score_pkl_path = luigi_pipeline.output()["score"].path

                luigi.build([luigi_pipeline], local_scheduler=True)

                with open(score_pkl_path, "rb") as in_file:
                    score = pickle.load(in_file)

                return score
            except:
                self.logger.debug("Pipeline evaluation failed!")
                return float("inf")
        self.logger.debug("Pipeline doesn't exists!")
        return float("inf")
