from ..template import FeaturePreprocessor


class NoFeaturePreprocessor(FeaturePreprocessor):
    abstract = False

    def output(self):
        return {
            "x_train": self.input()["scaled_features"]["x_train"],
            "x_test": self.input()["scaled_features"]["x_test"],
            "fitted_component": self.input()["scaled_features"]["fitted_component"]
        }
