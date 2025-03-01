import json
import os
from pathlib import Path
import wandb


import luigi
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from cls.debug_util import deep_str
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from lot_optimizers.groff_heuristic import GroffHeuristic
from lot_optimizers.least_unit_cost_method import LeastUnitCostMethod
from lot_optimizers.part_period_heuristic import PartPeriod
from lot_optimizers.silver_meal_heuristic import SilverMeal
from lot_optimizers.wagner_whitin import WagnerWhitin

from cls_luigi.inhabitation_task import ClsParameter, RepoMeta
from cls_luigi.utils.wandb import (
    WandbTask,
    run_luigi_pipeline_with_wandb,
    wandb_log,
    log_output,
    wandb_log_table,
    wandb_log_plot,
)

from cls_luigi.utils.wandb.config import 

class ConfigTask():
    prediction_horizon = luigi.IntParameter(default=8)


class GetCost(WandbTask):
    abstract = False

    def output(self):
        return [luigi.LocalTarget("data/cost.json")]

    def run(self):
        d = {
            "fixedCost": 400,  # Bestellkosten
            "varCost": 1,  # Lagerhaltungssatz
        }
        os.makedirs("data", exist_ok=True)
        with open(self.output()[0].path, "w") as f:
            json.dump(d, f, indent=4)


        wandb.log({"cost": d})
        wandb_log({"cost": d})

        wandb_log(
            {
                "cost": {
                    "path": self.output()[0].path,
                    "type": "dataset",
                    "metadata": {
                        "file_count": 1,
                        "file_size": os.path.getsize(self.output()[0].path),
                    },
                },
            }
        )


class GetHistoricDemand(WandbTask):
    abstract = False

    def output(self):
        print("GetHistoricDemand: output")
        return luigi.LocalTarget("data/historic_demand.csv")

    @log_output("dataset")
    def run(self):
        print("====== GetHistoricDemand: run")
        with self.output().open("w") as f:
            f.write(
                "1, 5, 7, 8, 9, 10, 14, 16, 19, 21, 19, 23, 24, 26, 26, "
                "26, 28, 26, 28, 30"
            )


class PredictDemand(WandbTask, ConfigTask):
    abstract = True
    get_historic_demand = ClsParameter(tpe=GetHistoricDemand.return_type())

    def requires(self):
        return {"historic_demand": self.get_historic_demand()}

    def _log_prediction_metrics(
        self,
        prediction_method,
        mae,
        mse,
        rmse,
        r2,
        mean_predicted_demand,
        std_predicted_demand,
        min_predicted_demand,
        max_predicted_demand,
    ):
        # Prepare metrics dictionary
        metrics = {
            f"{prediction_method}_mae": mae,
            f"{prediction_method}_mse": mse,
            f"{prediction_method}_rmse": rmse,
            f"{prediction_method}_r2": r2,
            f"{prediction_method}_mean_predicted_demand": mean_predicted_demand,
            f"{prediction_method}_std_predicted_demand": std_predicted_demand,
            f"{prediction_method}_min_predicted_demand": min_predicted_demand,
            f"{prediction_method}_max_predicted_demand": max_predicted_demand,
        }

        # Log metrics using wandb_log
        wandb_log(metrics, data_type="metric")

    def _log_prediction_plots(self, prediction_method, actual, predicted):
        # Prepare the prediction horizon
        prediction_horizon = list(
            range(len(actual))
        )  # Assuming actual and predicted have the same length

        # Prepare data for logging
        data = [[x, a, p] for x, a, p in zip(prediction_horizon, actual, predicted)]

        # Create a table for logging actual and predicted demands
        demand_table = wandb.Table(
            data=data,
            columns=["Prediction Horizon", "Actual Demand", "Predicted Demand"],
        )

        # Calculate residuals
        residuals = [a - p for a, p in zip(actual, predicted)]
        residuals_data = [[x, res] for x, res in zip(prediction_horizon, residuals)]

        # Create a table for logging residuals
        residuals_table = wandb.Table(
            data=residuals_data, columns=["Prediction Horizon", "Residuals"]
        )

        # Log tables using wandb_log_table
        wandb_log_table(demand_table, name=f"Demand Table for {prediction_method}")
        wandb_log_table(
            residuals_table, name=f"Residuals Table for {prediction_method}"
        )

        # Log plots using wandb_log_plot
        # Predicted Demand
        wandb_log_plot(
            wandb.plot.line(
                demand_table,
                "Prediction Horizon",
                "Predicted Demand",
                title=f"Predicted Demand for {prediction_method}",
            )
        )

        # Actual Demand
        wandb_log_plot(
            wandb.plot.line(
                demand_table,
                "Prediction Horizon",
                "Actual Demand",
                title=f"Actual Demand for {prediction_method}",
            )
        )

        # Actual and Predicted Demand in One Plot
        wandb_log_plot(
            wandb.plot.line_series(
                xs=prediction_horizon,
                ys=[list(actual), list(predicted)],
                keys=["Actual Demand", "Predicted Demand"],
                title=f"Actual and Predicted Demand for {prediction_method}",
                xname="Prediction Horizon",
            )
        )

        # Residuals
        wandb_log_plot(
            wandb.plot.scatter(
                residuals_table,
                "Prediction Horizon",
                "Residuals",
                title=f"Residuals for {prediction_method}",
            )
        )

        # Generate random data
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = np.random.randn(n)
        
        # Create a scatter plot with color and size variations
        fig = go.Figure(data=go.Scatter(
            x=x, 
            y=y, 
            mode='markers',
            marker=dict(
                size=10,
                color=x,  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                showscale=True
            ),
            text=[f'Point {i}' for i in range(n)],  # hover text
            hoverinfo='text'
        ))

        # Customize layout
        fig.update_layout(
            title='Random Scatter Plot',
            xaxis_title='X Values',
            yaxis_title='Y Values',
            template='plotly_white'
        )

        # Log the plot
        wandb_log_plot(fig, name='Random Scatter Visualization using Plotly')

    def track_prediction(
        self, prediction_method, actual, predicted, prediction_horizon=None
    ):
        # Log metrics
        actual = np.array(actual)
        predicted = np.array(predicted)

        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        mean_predicted_demand = np.mean(predicted)
        std_predicted_demand = np.std(predicted)
        min_predicted_demand = np.min(predicted)
        max_predicted_demand = np.max(predicted)

        self._log_prediction_metrics(
            prediction_method,
            mae,
            mse,
            rmse,
            r2,
            mean_predicted_demand,
            std_predicted_demand,
            min_predicted_demand,
            max_predicted_demand,
        )
        self._log_prediction_plots(prediction_method, actual, predicted)
        if prediction_horizon:
            wandb_log({"prediction_horizon": prediction_horizon}, data_type="params")

    def get_actual_demand(self):
        # just dummy values
        return [
            26,
            28,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
        ]

    def run(self):
        raise NotImplementedError()


class PredictDemandByLinearRegression(PredictDemand):
    abstract = False

    def output(self):
        return [luigi.LocalTarget("data/predicted_demand_by_linear_regression.pkl")]

    @log_output(
        lambda self: {
            "type": "model",
            "metadata": {
                "description": "Predicted demand DataFrame",
                "model_type": "linear_regression",
                "prediction_horizon": self.prediction_horizon,
            },
        }
    )
    def run(self):
        print("============= PredictDemandByLinearRegression: run")
        with self.input()["historic_demand"].open() as infile:
            print("I'm just a mock for Linear Regression!!!")
            predicted = [10 + i for i in range(self.prediction_horizon)]
            data = {"predicted_demand": predicted}
            df_predicted = pd.DataFrame(data)

            # Log metrics, hyperparameters and plots
            self.track_prediction(
                "linear regression",
                self.get_actual_demand()[: int(self.prediction_horizon)],
                predicted,
                int(self.prediction_horizon),
            )

            df_predicted.to_pickle(self.output()[0].path)


class PredictDemandByAverage(PredictDemand):
    abstract = False

    def output(self):
        return [luigi.LocalTarget("data/predicted_demand_by_average.pkl")]

    @log_output(
        lambda self: {
            "type": "model",
            "metadata": {
                "description": "Predicted demand by average",
                "model_type": "simple_average",
                "prediction_horizon": self.prediction_horizon,
            },
        }
    )
    def run(self):
        print("============= PredictDemandByAverage: run")
        with self.input()["historic_demand"].open() as infile:
            text = infile.read()
            l = [int(t) for t in text.split(",")]
            avg = int(sum(l) / len(l) + 0.5)
            predicted = [avg for i in range(self.prediction_horizon)]
            data = {"predicted_demand": predicted}
            df_predicted = pd.DataFrame(data)

            # Log metrics, hyperparameters and plots
            self.track_prediction(
                "average",
                self.get_actual_demand()[: int(self.prediction_horizon)],
                predicted,
                int(self.prediction_horizon),
            )

            df_predicted.to_pickle(self.output()[0].path)


class OptimizeLots(WandbTask, ConfigTask):
    """Base class for lot-sizing optimization tasks."""

    abstract = True
    predicted_demand = ClsParameter(tpe=PredictDemand.return_type())
    get_cost = ClsParameter(tpe=GetCost.return_type())

    def requires(self):
        return {"cost": self.get_cost(), "demand": self.predicted_demand()}

    def track_optimization(self, cost, metrics):
        self.optimization_cost = cost
        self.optimization_metrics = metrics

        # Log Hyperparameters using wandb_log
        wandb_log(
            {
                "planning_period": int(self.prediction_horizon),
                "fixed_cost": cost["fixedCost"],
                "variable_cost": cost["varCost"]
            }, 
            data_type="params"
        )

        # Log metrics using wandb_log
        wandb_log(metrics, data_type="metrics")

    def _get_cost(self):
        with open(self.input()["cost"][0].path, "rb") as f:
            cost = json.load(f)
        return cost

    def _get_demand(self):
        demand_df = pd.read_pickle(self.input()["demand"][0].path)
        return list(demand_df["predicted_demand"])

    @log_output(
        lambda self: {
            "type": "file",
            "metadata": {
                "description": "Lot optimization results",
                "optimizer": self.__class__.__name__,
                "demand_variant": self._get_variant_label(),
                "optimization_metrics": self.optimization_metrics,
            },
        }
    )
    def run(self):
        print(f"============= {self.__class__.__name__}: run")
        cost = self._get_cost()
        demand = self._get_demand()

        # Run optimizer
        orders, metrics = self.run_optimizer(cost, demand)

        self.track_optimization(cost, metrics)

        with self.output()[0].open("w") as f:
            f.write(str(list(orders)))

    def run_optimizer(self, cost, demand):
        return NotImplementedError()

    def _get_variant_label(self):
        if isinstance(self.input()["demand"][0], luigi.LocalTarget):
            label = self.input()["demand"][0].path
            return Path(label).stem


class OptimizeLotsByGroff(OptimizeLots):
    abstract = False

    def output(self):
        return [
            luigi.LocalTarget(
                "data/" + self._get_variant_label() + "-" + "optimize_lots_by_groff.txt"
            )
        ]

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByGroff: run_optimizer")
        optimizer = GroffHeuristic()
        orders, metrics = optimizer.run(cost, demand)

        return orders, metrics


class OptimizeLotsByWagnerWhitin(OptimizeLots):
    abstract = False

    def output(self):
        return [
            luigi.LocalTarget(
                "data/"
                + self._get_variant_label()
                + "-"
                + "optimize_lots_by_wagner_within.txt"
            )
        ]

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByWagnerWhitin: run_optimizer")
        optimizer = WagnerWhitin()
        orders, metrics = optimizer.run(cost, demand)

        return orders, metrics


class OptimizeLotsBySilverMeal(OptimizeLots):
    abstract = False

    def output(self):
        return [
            luigi.LocalTarget(
                "data/"
                + self._get_variant_label()
                + "-"
                + "optimize_lots_by_silver_meal.txt"
            )
        ]

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsBySilverMeal: run_optimizer")
        optimizer = SilverMeal()
        orders, metrics = optimizer.run(cost, demand)

        return orders, metrics


class OptimizeLotsByLeastUnitCost(OptimizeLots):
    abstract = False

    def output(self):
        return [
            luigi.LocalTarget(
                "data/"
                + self._get_variant_label()
                + "-"
                + "optimize_lots_by_least_unit_cost.txt"
            )
        ]

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByLeastUnitCost: run_optimizer")
        optimizer = LeastUnitCostMethod()
        orders, metrics = optimizer.run(cost, demand)

        return orders, metrics


class OptimizeLotsByPartPeriod(OptimizeLots):
    abstract = False

    def output(self):
        return [
            luigi.LocalTarget(
                "data/"
                + self._get_variant_label()
                + "-"
                + "optimize_lots_by_part_period.txt"
            )
        ]

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByPartPeriod: run_optimizer")
        optimizer = PartPeriod()
        orders, metrics = optimizer.run(cost, demand)

        return orders, metrics


if __name__ == "__main__":
    config = luigi.configuration.get_config()

    target = OptimizeLots.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    print(deep_str(inhabitation_result.rules))
    max_tasks_when_infinite = 50
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    if actual is not None or actual == 0:
        max_results = actual
    results = [t() for t in inhabitation_result.evaluated[0:max_results]]

    if results:
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        for pipeline in results[:1]:
            run_luigi_pipeline_with_wandb(pipeline, "lot_sizing", config=config)

    else:
        print("No results!")
