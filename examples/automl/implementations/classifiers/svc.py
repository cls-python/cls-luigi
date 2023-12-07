from ..template import Classifier
from sklearn.svm import SVC


class SKLKernelSVC(Classifier):
    abstract = False

    def run(self):
        self._read_split_processed_features()
        self._read_split_target_values()

        self.estimator = SVC(
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma=0.1,
            coef0=0.0,
            shrinking=True,
            tol=1e-3,
            max_iter=-1,
            random_state=self.global_params.seed,
            class_weight=None,
        )

        self.fit_predict_estimator()
        self.create_run_summary()
        self.sava_outputs()