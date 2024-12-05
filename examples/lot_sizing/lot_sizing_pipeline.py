import luigi
import os
from cls.debug_util import deep_str
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter
import pandas as pd
import json
from pathlib import Path
import numpy as np
import psutil

from lot_optimizers.groff_heuristic import GroffHeuristic
from lot_optimizers.wagner_whitin import WagnerWhitin
from lot_optimizers.silver_meal_heuristic import SilverMeal
from lot_optimizers.least_unit_cost_method import LeastUnitCostMethod
from lot_optimizers.part_period_heuristic import PartPeriod

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbTask(luigi.Task, LuigiCombinator):
    """Base class for tasks that use wandb logging."""

    enable_wandb = luigi.BoolParameter(default=False)
    prediction_horizon = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_run = None

    def get_historic_demand(self):
        """Get historic demand data."""
        with self.input()["historic_demand"].open() as infile:
            text = infile.read()
            return [int(t) for t in text.split(",")]

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

        # Log hyperparameters
        self.log_hyperparameters(
            {
                "fixed_cost": cost.get("fixedCost", 0),
                "variable_cost": cost.get("varCost", 0),
                "prediction_horizon": self.prediction_horizon,
                "optimizer": self.__class__.__name__,
                "total_demand": sum(demand) if demand else 0,
                "avg_demand": np.mean(demand) if demand else 0,
                "std_demand": np.std(demand) if demand else 0,
                "max_demand": max(demand) if demand else 0,
                "min_demand": min(demand) if demand else 0,
                "demand_length": len(demand) if demand else 0,
            }
        )

        # Log system metrics
        self.log_system_metrics()

        # Log plots and additional metrics
        if orders and demand:
            self.log_metrics_and_plots(orders, metrics, demand)

    def run(self):
        """Override this method in derived classes to implement task logic"""
        raise NotImplementedError()


class InitializeWandb(WandbTask):
    """Task to initialize wandb. Other tasks can depend on this to ensure wandb is initialized."""

    def output(self):
        return luigi.LocalTarget("data/wandb_initialized.txt")

    def run(self):
        if self.enable_wandb and WANDB_AVAILABLE:
            if not wandb.run:
                wandb.init(project="lot-sizing-optimization", entity=None, tags=[])

            # Save initialization info
            Path("data").mkdir(exist_ok=True)
            with self.output().open("w") as f:
                json.dump(
                    {
                        "project": "lot-sizing-optimization",
                        "entity": None,
                        "tags": [],
                        "run_id": wandb.run.id if wandb.run else None,
                    },
                    f,
                )
        else:
            # Create marker file even if wandb is disabled
            Path("data").mkdir(exist_ok=True)
            with self.output().open("w") as f:
                json.dump({"wandb_enabled": False}, f)


class GetCost(WandbTask):
    abstract = False

    def requires(self):
        if self.enable_wandb:
            return InitializeWandb(enable_wandb=self.enable_wandb)

    def output(self):
        return luigi.LocalTarget("data/cost.json")

    def run(self):
        d = {
            "fixedCost": 400,  # Bestellkosten
            "varCost": 1,  # Lagerhaltungssatz
        }
        os.makedirs("data", exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(d, f, indent=4)

        if self.enable_wandb:
            self.log_config(d)
            self.log_artifact(self.output().path, "cost_parameters", "parameters")


class GetHistoricDemand(WandbTask):
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
    prediction_horizon = 8

    def requires(self):
        reqs = {"historic_demand": self.get_historic_demand()}
        if self.enable_wandb:
            reqs["wandb"] = InitializeWandb(enable_wandb=self.enable_wandb)
        return reqs


class PredictDemandByLinearRegression(PredictDemand):
    abstract = False

    def output(self):
        return luigi.LocalTarget("data/predicted_demand_by_linear_regression.pkl")

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
            self.track_experiment(predicted, metrics, self.get_historic_demand(), {})

            self.plot_line_series(
                xs=[[i for i in range(len(predicted))]],
                ys=[predicted],
                keys=["Predicted Demand"],
                title="Predicted Demand over Time",
            )

            df_predicted.to_pickle(self.output().path)


class PredictDemandByAverage(PredictDemand):
    abstract = False

    def output(self):
        return luigi.LocalTarget("data/predicted_demand_by_average.pkl")

    def run(self):
        print("============= PredictDemandByAverage: run")
        with self.input()["historic_demand"].open() as infile:
            text = infile.read()
            l = [int(t) for t in text.split(",")]
            avg = int(sum(l) / len(l) + 0.5)
            predicted = [avg for _ in range(self.prediction_horizon)]
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
            self.track_experiment(predicted, metrics, self.get_historic_demand(), {})

            self.plot_line_series(
                xs=[[i for i in range(len(predicted))]],
                ys=[predicted],
                keys=["Predicted Demand"],
                title="Predicted Demand over Time",
            )

            df_predicted.to_pickle(self.output().path)


class OptimizeLots(WandbTask):
    abstract = True
    predicted_demand = ClsParameter(tpe=PredictDemand.return_type())
    get_cost = ClsParameter(tpe=GetCost.return_type())

    def requires(self):
        reqs = {"cost": self.get_cost(), "demand": self.predicted_demand()}
        if self.enable_wandb:
            reqs["wandb"] = InitializeWandb(enable_wandb=self.enable_wandb)
        return reqs

    def run(self):
        print(f"============= {self.__class__.__name__}: run")
        with self.input()["cost"].open() as cost_file:
            cost = json.load(cost_file)

        with self.input()["demand"].open("rb") as demand_file:
            df_demand = pd.read_pickle(demand_file)
            demand = df_demand["predicted_demand"].tolist()

        # Run optimizer
        orders, metrics = self.run_optimizer(cost, demand)
        self.track_experiment(orders, metrics, demand, cost)

    def calculate_inventory_levels(self, orders, demand):
        """Calculate inventory levels based on orders and demand"""
        inventory_levels = []
        current_inventory = 0
        for order, dem in zip(orders, demand):
            current_inventory += order - dem
            inventory_levels.append(current_inventory)
        return inventory_levels

    def run_optimizer(self, cost, demand):
        return NotImplementedError()

    def _get_variant_label(self):
        if isinstance(self.input()["demand"], luigi.LocalTarget):
            label = self.input()["demand"].path
            return Path(label).stem


class OptimizeLotsByGroff(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget(
            "data/" + self._get_variant_label() + "-" + "optimize_lots_by_groff.txt"
        )

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
        return luigi.LocalTarget(
            "data/"
            + self._get_variant_label()
            + "-"
            + "optimize_lots_by_wagner_within.txt"
        )

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
        return luigi.LocalTarget(
            "data/"
            + self._get_variant_label()
            + "-"
            + "optimize_lots_by_silver_meal.txt"
        )

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
        return luigi.LocalTarget(
            "data/"
            + self._get_variant_label()
            + "-"
            + "optimize_lots_by_least_unit_cost.txt"
        )

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByLeastUnitCost: run_optimizer")
        optimizer = LeastUnitCostMethod()
        orders = optimizer.run(cost, demand)
        
        metrics = {
            "total_cost": sum(orders),
            "fixed_costs": 0,
            "variable_costs": sum(orders),
            "cost_per_unit": self.metrics["total_cost"] / sum(demand) if sum(demand) > 0 else 0,
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
        return luigi.LocalTarget(
            "data/"
            + self._get_variant_label()
            + "-"
            + "optimize_lots_by_part_period.txt"
        )

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
            ys=[[metrics["fixed_costs"] for _ in range(len(orders))], [metrics["variable_costs"] for _ in range(len(orders))]],
            keys=["Fixed Costs", "Variable Costs"],
            title="Fixed and Variable Costs Over Time",
        )

        return orders, metrics


if __name__ == "__main__":

    target = OptimizeLots.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    print(deep_str(inhabitation_result.rules))
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    if not actual is None or actual == 0:
        max_results = actual
    results = [t() for t in inhabitation_result.evaluated[0:max_results]]

    if results:
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")
