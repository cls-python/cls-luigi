from ..template import CategoricalEncoder

from sklearn.preprocessing import OneHotEncoder


class OneHotEncoding(CategoricalEncoder):
    abstract = False

    def run(self):
        self.encoder = OneHotEncoder(
            categories="auto",
            handle_unknown="ignore",
            sparse_output=False
        )

        self._read_split_features()
        categorical_feature_names = self._get_categorical_features_names()
        if len(categorical_feature_names) > 0:
            self._fit_transform_encoder(categorical_feature_names, drop_original=True)
            self._save_outputs()
        else: 
            raise RuntimeError("No categorical features found, skipping encoding")
