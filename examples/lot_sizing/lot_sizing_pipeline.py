import luigi
from luigi import configuration
import os
from cls.debug_util import deep_str
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes

from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter


import pandas as pd
import statistics

import json
from pathlib import Path
import numpy as np
import psutil
import time

from lot_optimizers.groff_heuristic import GroffHeuristic
from lot_optimizers.wagner_whitin import WagnerWhitin
from lot_optimizers.silver_meal_heuristic import SilverMeal
from lot_optimizers.least_unit_cost_method import LeastUnitCostMethod
from lot_optimizers.part_period_heuristic import PartPeriod
import wandb

import re

# Optional wandb import
try:
    import wandb

    WANDB_IMPORTED = True
except ImportError:
    WANDB_IMPORTED = False


class WandbTask(luigi.Task, LuigiCombinator):
    """Base class for tasks that use wandb logging."""

    enable_wandb = luigi.BoolParameter(default=False)
    prediction_horizon = luigi.IntParameter(default=8)
    project_name = luigi.Parameter(default="lot_sizing")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = luigi.configuration.get_config()
        self.enable_wandb = config.getboolean(
            "WandbTask", "enable_wandb", self.enable_wandb
        )
        self.prediction_horizon = config.getint(
            "WandbTask", "prediction_horizon", self.prediction_horizon
        )
        self.project_name = config.get("WandbTask", "project_name", self.project_name)

    def run(self):
        """Override this method in derived classes to implement task logic"""
        raise NotImplementedError()

    @classmethod
    def wandb_init(cls, run_name=""):
        if not hasattr(cls, "wandb_instance"):
            instance = cls()
            cls.wandb_instance = wandb.init(project=str(instance.project_name), name=run_name)

    def plot_line_series(self, xs, ys, keys, title, xname="Period", yname="Value"):
        """Create a line plot in wandb."""
        if not self.enable_wandb:
            return

        data = []
        for x, y, key in zip(xs, ys, keys):
            for i, (xi, yi) in enumerate(zip(x, y)):
                data.append([xi, yi, key])

        table = wandb.Table(data=data, columns=[xname, yname, "Series"])
        wandb.log(
            {title: wandb.plot.line(table, xname, yname, title=title, stroke="Series")}
        )

    def log_if_enabled(self, log_function, *args, **kwargs):
        """Generalized logging method that checks if wandb is enabled."""
        if self.enable_wandb:
            log_function(*args, **kwargs)

    def log_metrics(self, metrics):
        """Log metrics to wandb."""
        self.log_if_enabled(wandb.log, metrics)

    def log_hyperparameters(self, params):
        """Log hyperparameters to wandb."""
        self.log_if_enabled(wandb.config.update, params)

    def log_system_metrics(self):
        """Log system metrics to wandb."""
        if not self.enable_wandb:
            return

        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available": psutil.virtual_memory().available,
            "memory_used": psutil.virtual_memory().used,
        }
        self.log_metrics(metrics)

    def log_metrics_and_plots(self, orders, metrics, demand):
        """Log metrics and create plots."""
        if not self.enable_wandb:
            return

        # Log order quantities over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[orders],
            keys=["Order Quantities"],
            title="Order Quantities over Time",
        )

        # Calculate and plot inventory levels
        inventory = self.calculate_inventory_levels(orders, demand)
        self.plot_line_series(
            xs=[[i for i in range(len(inventory))]],
            ys=[inventory],
            keys=["Inventory Levels"],
            title="Inventory Levels over Time",
        )

        # Plot demand vs orders
        self.plot_line_series(
            xs=[[i for i in range(len(demand))], [i for i in range(len(orders))]],
            ys=[demand, orders],
            keys=["Demand", "Orders"],
            title="Demand vs Orders over Time",
        )

    def track_experiment(self, orders, metrics, demand, cost):
        """Handles logging and tracking for wandb."""
        if not self.enable_wandb:
            return

        # Log metrics
        self.log_metrics(metrics)

        df = pd.read_csv(demand.path, header=None)

        demands = list(map(int, df.iloc[0, 0].split(',')))
        total_demand = sum(demands) if demands else 0
        avg_demand = total_demand / len(demands) if demands else 0
        std_demand = statistics.stdev(demands) if len(demands) > 1 else 0
        max_demand = max(demands) if demands else 0
        min_demand = min(demands) if demands else 0
        demand_length = len(demands) if demands else 0

        # Log hyperparameters
        self.log_hyperparameters(
            {
                "fixed_cost": cost.get("fixedCost", 0),
                "variable_cost": cost.get("varCost", 0),
                "prediction_horizon": self.prediction_horizon,
                "optimizer": self.__class__.__name__,
                "total_demand": total_demand,
                "avg_demand": avg_demand,
                "std_demand": std_demand,
                "max_demand": max_demand,
                "min_demand": min_demand,
                "demand_length": demand_length,
            }
        )

        # Log system metrics
        self.log_system_metrics()

        # Log plots and additional metrics
        if orders and demand:
            self.log_metrics_and_plots(orders, metrics, demand)

    def calculate_inventory_levels(self, orders, demand):
        """Calculate inventory levels based on orders and demand"""
        inventory_levels = []
        current_inventory = 0
        for order, dem in zip(orders, demand):
            current_inventory += order - dem
            inventory_levels.append(current_inventory)
        return inventory_levels


class InitializeWandb(WandbTask):
    """Task to initialize wandb for the entire pipeline."""

    abstract = False
    pipeline_name = luigi.Parameter(default="none")

    def complete(self):
        return True

    def run(self):
        if self.enable_wandb and WANDB_IMPORTED:
            self.wandb_init(run_name=self.pipeline_name + "_" + time.strftime("%Y%m%d-%H%M%S"))


class GetCost(WandbTask):
    abstract = False

    def requires(self):
        return InitializeWandb()

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

        # if self.enable_wandb:
        #     self.log_config(d)
        #     self.log_artifact(self.output().path, "cost_parameters", "parameters")


class GetHistoricDemand(WandbTask):
    abstract = False

    def requires(self):
        return InitializeWandb()

    def output(self):
        print("GetHistoricDemand: output")
        return luigi.LocalTarget("data/historic_demand.csv")

    def run(self):
        print("====== GetHistoricDemand: run")
        with self.output().open("w") as f:
            f.write(
                "1, 5, 7, 8, 9, 10, 14, 16, 19, 21, 19, 23, 24, 26, 26, "
                "26, 28, 26, 28, 30"
            )


class PredictDemand(WandbTask):
    abstract = True
    get_historic_demand = ClsParameter(tpe=GetHistoricDemand.return_type())

    def requires(self):
        return {"historic_demand": self.get_historic_demand()}

    def run(self):
        raise NotImplementedError()


class PredictDemandByLinearRegression(PredictDemand):
    abstract = False

    def output(self):
        return [luigi.LocalTarget("data/predicted_demand_by_linear_regression.pkl")]

    def run(self):
        print("============= PredictDemandByLinearRegression: run")
        with self.input()["historic_demand"].open() as infile:
            print("I'm just a mock for Linear Regression!!!")
            predicted = [10 + i for i in range(self.prediction_horizon)]
            data = {"predicted_demand": predicted}
            df_predicted = pd.DataFrame(data)

            # Log metrics and plots
            metrics = {
                "prediction_method": "linear_regression",
                "mean_predicted_demand": np.mean(predicted),
                "std_predicted_demand": np.std(predicted),
                "min_predicted_demand": min(predicted),
                "max_predicted_demand": max(predicted),
            }
            self.track_experiment(predicted, metrics, self.input()["historic_demand"].path, {})

            self.plot_line_series(
                xs=[[i for i in range(len(predicted))]],
                ys=[predicted],
                keys=["Predicted Demand"],
                title="Predicted Demand over Time",
            )

            df_predicted.to_pickle(self.output()[0].path)


class PredictDemandByAverage(PredictDemand):
    abstract = False

    def output(self):
        return [luigi.LocalTarget("data/predicted_demand_by_average.pkl")]

    def run(self):
        print("============= PredictDemandByAverage: run")
        with self.input()["historic_demand"].open() as infile:
            text = infile.read()
            l = [int(t) for t in text.split(",")]
            avg = int(sum(l) / len(l) + 0.5)
            predicted = [avg for i in range(self.prediction_horizon)]
            data = {"predicted_demand": predicted}
            df_predicted = pd.DataFrame(data)

            # Log metrics and plots
            metrics = {
                "prediction_method": "average",
                "mean_predicted_demand": np.mean(predicted),
                "std_predicted_demand": np.std(predicted),
                "min_predicted_demand": min(predicted),
                "max_predicted_demand": max(predicted),
            }
            self.track_experiment(predicted, metrics, predicted, {})

            self.plot_line_series(
                xs=[[i for i in range(len(predicted))]],
                ys=[predicted],
                keys=["Predicted Demand"],
                title="Predicted Demand over Time",
            )

            df_predicted.to_pickle(self.output()[0].path)


class OptimizeLots(WandbTask):
    """Base class for lot-sizing optimization tasks."""

    abstract = True
    predicted_demand = ClsParameter(tpe=PredictDemand.return_type())
    get_cost = ClsParameter(tpe=GetCost.return_type())

    def requires(self):
        return {"cost": self.get_cost(), "demand": self.predicted_demand()}

    def _get_cost(self):
        with open(self.input()["cost"][0].path, "rb") as f:
            cost = json.load(f)
        return cost

    def _get_demand(self):
        demand_df = pd.read_pickle(self.input()["demand"][0].path)
        return list(demand_df["predicted_demand"])

    def run(self):
        print(f"============= {self.__class__.__name__}: run")
        cost = self._get_cost()
        demand = self._get_demand()

        # Run optimizer
        orders, metrics = self.run_optimizer(cost, demand)
        self.track_experiment(orders, metrics, demand, cost)

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
        orders = optimizer.run(cost, demand)

        metrics = {
            "total_cost": sum(orders),
            "fixed_costs": 0,
            "variable_costs": sum(orders),
            "criterion_value": (2 * cost["fixedCost"]) / cost["varCost"],
        }
        self.track_experiment(orders, metrics, demand, cost)

        # Plot order quantities over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[orders],
            keys=["Order Quantities"],
            title="Order Quantities Over Time",
        )

        # Plot total cost over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[[metrics["total_cost"] for _ in range(len(orders))]],
            keys=["Total Cost"],
            title="Total Cost Over Time",
        )

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
        orders = optimizer.run(cost, demand)

        metrics = {
            "total_cost": sum(orders),
            "fixed_costs": 0,
            "variable_costs": sum(orders),
            "num_orders": len(orders),
        }
        self.track_experiment(orders, metrics, demand, cost)

        # Plot order quantities over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[orders],
            keys=["Order Quantities"],
            title="Order Quantities Over Time",
        )

        # Plot total cost over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[[metrics["total_cost"] for _ in range(len(orders))]],
            keys=["Total Cost"],
            title="Total Cost Over Time",
        )

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
        orders = optimizer.run(cost, demand)

        metrics = {
            "total_cost": sum(orders),
            "fixed_costs": 0,
            "variable_costs": sum(orders),
            "avg_order_quantity": np.mean(orders),
        }
        self.track_experiment(orders, metrics, demand, cost)

        # Plot order quantities over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[orders],
            keys=["Order Quantities"],
            title="Order Quantities Over Time",
        )

        # Plot average order quantity over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[[metrics["avg_order_quantity"] for _ in range(len(orders))]],
            keys=["Average Order Quantity"],
            title="Average Order Quantity Over Time",
        )

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
        orders = optimizer.run(cost, demand)

        metrics = {
            "total_cost": sum(orders),
            "fixed_costs": 0,
            "variable_costs": sum(orders),
            "cost_per_unit": 0 / sum(demand) if sum(demand) > 0 else 0,
        }
        self.track_experiment(orders, metrics, demand, cost)

        # Plot order quantities over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[orders],
            keys=["Order Quantities"],
            title="Order Quantities Over Time",
        )

        # Plot cost per unit over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[[metrics["cost_per_unit"] for _ in range(len(orders))]],
            keys=["Cost per Unit"],
            title="Cost per Unit Over Time",
        )

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
        orders = optimizer.run(cost, demand)

        metrics = {
            "total_cost": sum(orders),
            "fixed_costs": 0,
            "variable_costs": sum(orders),
        }
        self.track_experiment(orders, metrics, demand, cost)

        # Plot order quantities over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[orders],
            keys=["Order Quantities"],
            title="Order Quantities Over Time",
        )

        # Plot fixed and variable costs over time
        self.plot_line_series(
            xs=[[i for i in range(len(orders))]],
            ys=[
                [metrics["fixed_costs"] for _ in range(len(orders))],
                [metrics["variable_costs"] for _ in range(len(orders))],
            ],
            keys=["Fixed Costs", "Variable Costs"],
            title="Fixed and Variable Costs Over Time",
        )

        return orders, metrics


class FinalizeWandb(WandbTask):
    """Task to finalize wandb logging."""

    abstract = False
    target_task = ClsParameter(tpe=OptimizeLots.return_type())

    def requires(self):
        return self.target_task()

    def complete(self):
        return True

    def run(self):
        if self.enable_wandb:
            wandb.finish()


def extract_task_classes(input_str):
    task_classes = []

    main_class_pattern = r"(\w+)\("
    main_class_match = re.search(main_class_pattern, input_str)
    if main_class_match:
        task_classes.append(main_class_match.group(1))  # Get the class name
    pattern = r'"task_class":\s*"([^"]+)"'
    matches = re.findall(pattern, input_str)
    task_classes.extend(matches)
    return "_".join(task_classes)


if __name__ == "__main__":

    # Set global configuration for all tasks
    config = configuration.get_config()
    config.set("WandbTask", "enable_wandb", "True")
    config.set("WandbTask", "prediction_horizon", "5")
    config.set("WandbTask", "project_name", "TestProject")

    target = FinalizeWandb.return_type()
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
        for r in results:
            config.set(
                "InitializeWandb", "pipeline_name", f"{extract_task_classes(str (r))}"
            )
            pipeline = r
            print(type(pipeline))
            print("==============")
            print(pipeline)
            print("\n")
            #luigi.build([r], local_scheduler=True, detailed_summary=True)
    else:
        print("No results!")
