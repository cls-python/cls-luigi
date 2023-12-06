from ..template import Classifier
from sklearn.ensemble import ExtraTreesClassifier


class SKLExtraTrees(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        max_features = int(self.x_train.shape[1] ** 0.5)
        if max_features == 0:
            max_features = "sqrt"
        self.estimator = ExtraTreesClassifier(
            criterion="gini",
            max_features=max_features,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=False,
            random_state=self.global_params.seed
        )

        self.fit_predict_estimator()
        self.create_run_summary()
        self.sava_outputs()
