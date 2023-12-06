from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from ..template import Classifier



class SKLQuadraticDiscriminantAnalysis(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        self.estimator = QuadraticDiscriminantAnalysis(
            reg_param=0.0,
        )

        self.fit_predict_estimator()
        self.create_run_summary()
        self.sava_outputs()
