from sklearn.model_selection import train_test_split
from ..template import LoadAndSplitData
import pandas as pd


class LoadAndSplitPickledTabularData(LoadAndSplitData):
    abstract = False

    def run(self):
        
        features_path = self.global_params.X_path
        labels_path = self.global_params.y_path
        
        X = pd.read_pickle(features_path)
        y = pd.read_pickle(labels_path)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
    

        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        

        x_train.to_pickle(self.output()["x_train"].path)
        x_test.to_pickle(self.output()["x_test"].path)
        y_train.to_pickle(self.output()["y_train"].path)
        y_test.to_pickle(self.output()["y_test"].path)

