from ..template import PredictDemand
import pandas as pd
import luigi


class PredictDemandByAverage(PredictDemand):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/predicted_demand_by_average.pkl')

    def run(self):
        print("============= PredictDemandByAverage: run")
        with self.input().open() as infile:
            text = infile.read()
            l = [int(t) for t in text.split(",")]
            avg = int(sum(l) / len(l) + 0.5)
            predicted = [avg for i in range(self.prediction_horizon)]
            data = {'predicted_demand': predicted}
            df_predicted = pd.DataFrame(data)

            df_predicted.to_pickle(self.output().path)
