from ..template import FeaturePreprocessor
from sklearn.ensemble import RandomTreesEmbedding
from examples.automl.utils.time_recorder import TimeRecorder
import warnings


class SKLRandomTreesEmbedding(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_scaled_features()

                self.feature_preprocessor = RandomTreesEmbedding(
                    n_estimators=10,
                    max_depth=5,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0,  # TODO: check
                    max_leaf_nodes=None,
                    n_jobs=self.global_params.n_jobs,
                    random_state=self.global_params.seed,
                    sparse_output=False
                )

                self.fit_transform_feature_preprocessor(x_and_y_required=False, handle_sparse_output=False)
                self.sava_outputs()
