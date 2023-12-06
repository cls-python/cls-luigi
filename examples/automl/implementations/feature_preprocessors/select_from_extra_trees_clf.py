from ..template import FeaturePreprocessor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


class SKLSelectFromExtraTrees(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()
        self._read_split_target_values()

        estimator = ExtraTreesClassifier(
            n_estimators=100,
            criterion="gini",
            max_features=0.5,
            max_depth=None,
            max_leaf_nodes=None,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            min_impurity_decrease=0.0,
            bootstrap=False,
            random_state=self.global_params.seed,
            oob_score=False,
            n_jobs=self.global_params.n_jobs,
            verbose=0
        )

        estimator.fit(self.x_train, self.y_train)
        self.feature_preprocessor = SelectFromModel(
            estimator=estimator,
            threshold="mean",
            prefit=True
        )
        self.fit_transform_feature_preprocessor(x_and_y_required=True)
        self.sava_outputs()
