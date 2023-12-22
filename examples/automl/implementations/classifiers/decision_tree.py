from ..template import Classifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import warnings
from examples.automl.utils.time_recorder import TimeRecorder

class SKLDecisionTree(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()
                self._read_split_target_values()

                num_features = self.x_train.shape[1]

                max_depth_factor = max(
                    1, int(np.round(0.5 * num_features, 0))
                )

                self.estimator = DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=max_depth_factor,  #
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_features=1.0,
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.0,
                    random_state=self.global_params.seed,
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
