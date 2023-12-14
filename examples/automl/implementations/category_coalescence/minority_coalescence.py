from ..template import CategoryCoalescer
from .minority_coalescer_implementation import MinorityCoalescer



class MinorityCoalescence(CategoryCoalescer):
    abstract = False

    def run(self):

        self.component = MinorityCoalescer(minimum_fraction=0.01)
        self._read_split_features()
        categorical_features_names = self._get_categorical_features_names()

        self.component.fit(self.x_train[categorical_features_names].values)

        self.x_train[categorical_features_names] = self.component.transform(self.x_train[categorical_features_names].values)
        self.x_train[categorical_features_names] = self.x_train[categorical_features_names].astype("category")
        self.x_test[categorical_features_names] = self.component.transform(self.x_test[categorical_features_names].values)
        self.x_test[categorical_features_names] = self.x_test[categorical_features_names].astype("category")

        self._save_outputs()

        




