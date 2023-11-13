import os.path
import numpy as np
from openml import tasks

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_dataset(task_id):
    task = tasks.get_task(task_id)
    X, y = task.get_X_and_y(dataset_format='dataframe')
    d_name = task.get_dataset().name

    return X, y, d_name


def drop_unnamed_col(df):
    unnamed_col = "Unnamed: 0"

    if unnamed_col in list(df.columns):
        return df.drop([unnamed_col], axis=1)
    return df


def encode_labels(y):
    classes = sorted(list(y.unique()))
    print("Original classes: ", classes)

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

    new_classes = sorted(list(y.unique()))
    print("Encoded classes: ", new_classes)
    print(classes)
    return y


if __name__ == "__main__":

    DATASET_DIR = "datasets"
    if os.path.exists(DATASET_DIR) is False:
        os.mkdir(DATASET_DIR)

    openml_tasks = [
        146820,  # wilt classification
        168868,  # APSFailure classification
        168911,  # jasmine classification
        168350,  # phoneme classification
        359958,  # pc4 classification
        359962,  # kc1 classification
        359972,  # sylvin classification
        359990,  # MiniBooNE classification
    ]



    for t in openml_tasks:
        print("Getting dataset: ", t)
        X, y, d_name = get_dataset(t)
        print("Dataset name: ", d_name)
        print("Dataset types: ", X.dtypes)
        print("Dataset shape: ", X.shape)
        print("y dtypes: ", y.dtypes)
        print("y unique: ", sorted(list(y.unique())))
        y = encode_labels(y)
        print("y dtypes: ", y.dtypes)
        print("y unique: ", y.unique())
        print("=====================================\n")

        X = drop_unnamed_col(X)
        joined = X.copy()
        joined["target"] = y.tolist()
        joined.to_csv(os.path.join(DATASET_DIR, d_name + ".csv"), index=False)
