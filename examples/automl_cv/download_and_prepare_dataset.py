import pickle

from sklearn import model_selection, datasets
import os


def load_breast_cancer_dataset():
    x, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    return x, y


def split_dataset(X, y,
                  test_size=0.2,
                  random_state=42):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state)

    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return x_train, x_test, y_train, y_test


def generate_folds(x_train,
                   y_train,
                   n_splits=3,
                   random_state=42,
                   shuffle=True):
    kf = model_selection.KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state)

    folds = {}

    for fold, (training_index, validation_index) in enumerate(kf.split(x_train)):
        folds[fold] = {
            "training": {
                "x": x_train.loc[training_index].reset_index(drop=True),
                "y": y_train.loc[training_index].reset_index(drop=True)
            },
            "validation": {
                "x": x_train.loc[validation_index].reset_index(drop=True),
                "y": y_train.loc[validation_index].reset_index(drop=True)
            }
        }
    return folds


def save_dataset(training_folds, x_train, y_train, x_test, y_test, path):
    os.makedirs(path, exist_ok=True)

    x_train.to_pickle(os.path.join(path, "x_train.pkl"))
    y_train.to_pickle(os.path.join(path, "y_train.pkl"))
    x_test.to_pickle(os.path.join(path, "x_test.pkl"))
    y_test.to_pickle(os.path.join(path, "y_test.pkl"))

    x_test.to_pickle(os.path.join(path, "x_infer.pkl"))


    with open(os.path.join(path, "cv_features_and_targets.pkl"), "wb") as f:
        pickle.dump(training_folds, f)


if __name__ == "__main__":
    X, y = load_breast_cancer_dataset()
    x_train, x_test, y_train, y_test = split_dataset(X, y)
    training_folds = generate_folds(x_train, y_train)
    save_dataset(training_folds, x_train, y_train, x_test, y_test, "datasets/breast_cancer")
