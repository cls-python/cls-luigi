from ..template import PredictDemand
import pandas as pd
import luigi


class PredictDemandByLinearRegression(PredictDemand):
    abstract = False

    def output(self):
        return luigi.LocalTarget(
            'data/predicted_demand_by_linear_regression.pkl')

    def run(self):
        print("============= PredictDemandByLinearRegression: run")
        with self.input().open() as infile:
            print("I'm just a mock for Linear Regression!!!")
            predicted = [10 + i for i in range(self.prediction_horizon)]
            data = {'predicted_demand': predicted}
            df_predicted = pd.DataFrame(data)

            df_predicted.to_pickle(self.output().path)
