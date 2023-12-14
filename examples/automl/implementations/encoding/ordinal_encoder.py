from ..template import CategoricalEncoder
from sklearn.preprocessing import OrdinalEncoder


class OrdinalEncoding(CategoricalEncoder):
    abstract = False

    def run(self):
        self.encoder = OrdinalEncoder(#
            categories="auto",
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        self._read_split_features()
        categorical_feature_names = self._get_categorical_features_names()
        if len(categorical_feature_names) > 0:
            self._fit_transform_encoder(categorical_feature_names, drop_original=False, suffix="_ordinal") # suffix must be used, otherwise we get a duplicate column name
            self._save_outputs()
        else: 
            raise RuntimeError("No categorical features found, skipping encoding")
