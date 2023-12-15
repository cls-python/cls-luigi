import numpy as np
from openml import tasks
from sklearn.model_selection import train_test_split

from ..template import LoadAndSplitData


class LoadOpenMLDataAndSplit(LoadAndSplitData):
    abstract = False

    def run(self):

        X, y, d_name = self._get_openml_dataset(self.global_params.dataset_name)
        y = self._encode_labels(y)
        X = self._drop_unnamed_col(X)

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

    def _get_openml_dataset(self, task_id):
        task = tasks.get_task(task_id)
        X, y = task.get_X_and_y(dataset_format='dataframe')
        d_name = task.get_dataset().name

        return X, y, d_name

    @staticmethod
    def _encode_labels(y):
        classes = sorted(list(y.unique()))

        if isinstance(classes[0], str) and isinstance(classes[1], str):
            if classes[0].isnumeric() and classes[1].isnumeric():
                y = y.map(lambda x: 0 if x == classes[0] else 1)
            elif classes[0].isnumeric() is False and classes[1].isnumeric() is False:
                y = y.map(lambda x: 0 if x == "neg" else 1)
            else:
                raise TypeError("Label is string but neither numeric or neg/pos")
        elif (isinstance(classes[0], bool) and isinstance(classes[1], bool)) or \
            (isinstance(classes[0], np.bool_) and isinstance(classes[1], np.bool_)):
            y = y.map(lambda x: 0 if x == False else 1)

        else:
            raise TypeError("Label is not string nor bool")

        return y

    @staticmethod
    def _drop_unnamed_col(df):
        unnamed_col = "Unnamed: 0"

        if unnamed_col in list(df.columns):
            return df.drop([unnamed_col], axis=1)
        return df
