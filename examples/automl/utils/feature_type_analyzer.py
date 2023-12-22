import pandas as pd


class FeatureTypeAnalyzer(object):

    def __init__(self, csv_file_path: str) -> None:
        self.csv_file_path = csv_file_path
        self.dataframe = pd.read_csv(csv_file_path)

    def has_categorical_features(self) -> bool:
        categorical_feature_names = self.dataframe.select_dtypes(include=['category']).columns.tolist()

        if len(categorical_feature_names) > 0:
            return True
        else:
            return False

    def has_string_features(self) -> bool:
        string_feature_names = self.dataframe.select_dtypes(include=['object']).columns.tolist()

        if len(string_feature_names) > 0:
            return True
        else:
            return False

    def has_numerical_features(self) -> bool:
        numerical_feature_names = self.dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if len(numerical_feature_names) > 0:
            return True
        else:
            return False
