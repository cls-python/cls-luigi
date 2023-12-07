from sklearn.linear_model import PassiveAggressiveClassifier
from ..template import Classifier

class SKLPassiveAggressive(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        self.estimator = PassiveAggressiveClassifier(
            C=1.0,
            fit_intercept=True,
            loss="hinge",
            tol=1e-4,
            average=False,
            shuffle=True,
            random_state=self.global_params.seed,
            warm_start=True
        )

        self.fit_predict_estimator()
        self.create_run_summary()
        self.sava_outputs()
