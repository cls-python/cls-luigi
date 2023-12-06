from ..template import Scaler


class NoScaling(Scaler):
    abstract = False

    def output(self):
        return {
            "x_train": self.input()["x_train"],
            "x_test": self.input()["x_test"],
            "fitted_component": self.input()["fitted_component"]
        }
