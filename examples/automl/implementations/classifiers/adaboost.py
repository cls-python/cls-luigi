import warnings

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ..template import Classifier
from examples.automl.utils.time_recorder import TimeRecorder


class SKLAdaBoost(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_target_values()
                self._read_split_processed_features()

                base_estimator = DecisionTreeClassifier(max_depth=1, random_state=self.global_params.seed)

                self.estimator = AdaBoostClassifier(
                    estimator=base_estimator,
                    n_estimators=50,
                    learning_rate=1.0,
                    algorithm="SAMME.R",
                    random_state=self.global_params.seed
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
