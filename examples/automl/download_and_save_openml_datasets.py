import numpy as np
from openml import tasks
import os


def download_and_save_openml_dataset(dataset_id):
    
    X, y, ds_name = _get_openml_dataset(dataset_id)
    y = _encode_classification_labels(y)
    X = _drop_unnamed_col(X)
    
    dataset_dir = f"datasets/{ds_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    X_path = os.path.join(dataset_dir, "X.pkl")
    y_path = os.path.join(dataset_dir, "y.pkl")

    X.to_pickle(X_path)
    y.to_pickle(y_path)
        
    return X_path, y_path, ds_name
    
    
    
def _get_openml_dataset(task_id):
    task = tasks.get_task(task_id)
    X, y = task.get_X_and_y(dataset_format='dataframe')
    d_name = task.get_dataset().name

    return X, y, d_name

def _encode_classification_labels(y):
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

def _drop_unnamed_col(df):
    unnamed_col = "Unnamed: 0"

    if unnamed_col in list(df.columns):
        return df.drop([unnamed_col], axis=1)
    return df




if __name__ == "__main__":
    
    task_list = [
    361066,  # bank-marketing classification
    #146820,  # wilt classification
    168868,  # APSFailure classification
    168911,  # jasmine classification
    # 168350,  # phoneme classification contains negative values
    # 359958,  # pc4 classification
    # 359962,  # kc1 classification
    # 359972,  # sylvin classification
    #359990,  # MiniBooNE classification
    # 146606,  #higgs
    ]
    
    for task_id in task_list:
        download_and_save_openml_dataset(task_id)