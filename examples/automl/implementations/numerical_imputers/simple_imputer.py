from ..template import NumericalImputer
from sklearn.impute import SimpleImputer


class SKLSimpleImpute(NumericalImputer):
    abstract = False

    def run(self):
        self._read_split_features_from_input()

        self.imputer = SimpleImputer(
            strategy="mean",
            copy=False
        )

        self._fit_transform_imputer()
        self._save_outputs()
