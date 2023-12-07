from sklearn.svm import LinearSVC
from ..template import Classifier


class SKLLinearSVC(Classifier):
    abstract = False

    def run(self):
        self._read_split_processed_features()
        self._read_split_target_values()

        self.estimator = LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual=False,
            tol=1e-4,
            C=1.0,
            multi_class="ovr",
            fit_intercept=True,
            intercept_scaling=1,
            random_state=self.global_params.seed
        )

        self.fit_predict_estimator()
        self.create_run_summary()
        self.sava_outputs()
