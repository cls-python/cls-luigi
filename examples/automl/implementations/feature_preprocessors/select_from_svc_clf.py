from ..template import FeaturePreprocessor
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


class SKLSelectFromLinearSVC(FeaturePreprocessor):
    abstract = False

    def run(self):
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
