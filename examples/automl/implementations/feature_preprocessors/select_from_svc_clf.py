from ..template import FeaturePreprocessor
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from examples.automl.utils.time_recorder import TimeRecorder
import warnings

class SKLSelectFromLinearSVC(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_scaled_features()
                self._read_split_target_values()

                estimator = LinearSVC(
                    penalty="l1",
                    loss="squared_hinge",
                    dual=False,
                    tol=1e-4,
                    C=1.0,
                    multi_class="ovr",
                    fit_intercept=True,
                    intercept_scaling=1,
                    random_state=self.global_params.seed
                )
                estimator.fit(self.x_train, self.y_train)
                self.feature_preprocessor = SelectFromModel(
                    estimator=estimator,
                    threshold="mean",
                    prefit=True
                )

                self.fit_transform_feature_preprocessor(x_and_y_required=True)
                self.sava_outputs()
