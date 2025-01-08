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
import threading
import re

from lot_optimizers.groff_heuristic import GroffHeuristic
from lot_optimizers.wagner_whitin import WagnerWhitin
from lot_optimizers.silver_meal_heuristic import SilverMeal
from lot_optimizers.least_unit_cost_method import LeastUnitCostMethod
from lot_optimizers.part_period_heuristic import PartPeriod

# Optional GPUtil for nvidia gpu usage
try:
    import GPUtil
    NVIDIAGPU_IMPORTED = True
except ImportError:
    NVIDIAGPU_IMPORTED = False

# Optional pyamdgpuinfo for amd gpu usage
try:
    import pyamdgpuinfo
    AMDGPU_IMPORTED = True
except ImportError:
    AMDGPU_IMPORTED = False

# Optional wandb import
try:
    import wandb
    WANDB_IMPORTED = True
except ImportError:
    WANDB_IMPORTED = False

class ConfigTask(luigi.Task):
    enable_wandb = luigi.BoolParameter(default=True)
    prediction_horizon = luigi.IntParameter(default=8)

class WandbTask(ConfigTask, LuigiCombinator):
    """Base class for tasks that use wandb logging."""

    abstract = True

    def run(self):
        """Override this method in derived classes to implement task logic"""
        raise NotImplementedError()

    def on_failure(self, exception):
        if self.enable_wandb:
            if wandb.run is not None:
                wandb.log({"status": "failed", "error": str(exception)})
                wandb.finish()
        return super().on_failure(exception)

    def log_artifact(self, file_path, artifact_name, artifact_type='dataset', log_to_run=True):
        """Log an artifact to WandB, either to a specific run or to the project."""
        if self.enable_wandb:
            if log_to_run and wandb.run is not None:
                # Log to the current run
                artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
                artifact.add_file(file_path)
                wandb.log_artifact(artifact)
                print(f"Logged {artifact_type} artifact to run: {artifact_name}")
            else:
                # Log to the project without associating with a run
                api = wandb.Api()
                artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata={"project": str(self.project_name)})
                artifact.add_file(file_path)
                api.artifacts.create(artifact)
                print(f"Logged {artifact_type} artifact to project: {artifact_name}")

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

    def log_prediction_metrics(mae, mse, rmse, r2, mean_predicted_demand, std_predicted_demand, min_predicted_demand, max_predicted_demand):
        wandb.log({
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R-squared": r2,
            "Mean Predicted Demand": mean_predicted_demand,
            "Std Predicted Demand": std_predicted_demand,
            "Min Predicted Demand": min_predicted_demand,
            "Max Predicted Demand": max_predicted_demand,
        })

    def log_prediction_plots(actual, predicted):
        # Log Actual vs Predicted
        wandb.log({
            "Actual vs Predicted": wandb.plot.line(
                x=list(range(len(actual))), 
                y=actual, 
                title="Actual Demand",
                xname="Time",
                yname="Demand"
            )
        })
        
        wandb.log({
            "Predicted vs Actual": wandb.plot.line(
                x=list(range(len(predicted))), 
                y=predicted, 
                title="Predicted Demand",
                xname="Time",
                yname="Demand"
            )
        })

        # Log Residuals
        residuals = actual - predicted
        wandb.log({
            "Residuals": wandb.plot.scatter(
                x=predicted, 
                y=residuals, 
                title="Residual Plot",
                xname="Predicted Demand",
                yname="Residuals"
            )
        })

        # Demand over Time
        wandb.log({
            "Demand Over Time": wandb.plot.line(
                x=list(range(len(actual))), 
                y=actual, 
                title="Demand Over Time",
                xname="Time",
                yname="Demand"
            )
        })

        # Predicted Demand over Time
        wandb.log({
            "Predicted Demand Over Time": wandb.plot.line(
                x=list(range(len(predicted))), 
                y=predicted, 
                title="Predicted Demand Over Time",
                xname="Time",
                yname="Demand"
            )
        })

    def track_prediction(self, prediction_method, actual, predicted):
        if not self.enable_wandb:
            return

        # Log metrics
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

        self.log_prediction_metrics(mae, mse, rmse, r2, mean_predicted_demand, std_predicted_demand, min_predicted_demand, max_predicted_demand)
        self.log_prediction_plots(actual, predicted)
   
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

        # Log plots and additional metrics
        if orders and demand:
            self.log_metrics_and_plots(orders, metrics, demand)

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

        if self.enable_wandb:
            self.log_artifact(
                self.output()[0].path,
                artifact_name="cost",
                artifact_type="dataset",
                log_to_run=False,
            )


class GetHistoricDemand(WandbTask):
    abstract = False

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
        if self.enable_wandb:
            self.log_artifact(
                self.output().path,
                artifact_name="historic_demand",
                artifact_type="dataset",
                log_to_run=False,
            )


class PredictDemand(WandbTask):
    abstract = True
    get_historic_demand = ClsParameter(tpe=GetHistoricDemand.return_type())

    def requires(self):
        return {"historic_demand": self.get_historic_demand(), "actual_demand": self.get_actual_demand()}

    def get_actual_demand(self):
        # just dummy values
        return [26, 28, 26, 28, 30, 31, 32, 33, 34, 35 , 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

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
            self.track_prediction("linear regression", self.get_actual_demand(), predicted)

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
            self.track_prediction("average", self.get_actual_demand(), predicted)

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

# Create a global event to control the logging thread
stop_event = threading.Event()

def reset_stop_event():
    global stop_event
    stop_event = threading.Event()  # Create a new event, which is unset (False)

def get_nvidia_usage():
    """Get GPU usage for NVIDIA and AMD GPUs."""
    gpu_usage = {}

    # Check for NVIDIA GPUs
    nvidia_gpus = GPUtil.getGPUs()
    if nvidia_gpus:
        gpu_usage['NVIDIA'] = [gpu.load * 100 for gpu in nvidia_gpus]

    return gpu_usage

def get_amd_usage():
    """Get GPU usage for AMD GPUs."""
    try:
        amd_gpus = pyamdgpuinfo.get_all_gpus()  # Get information about all AMD GPUs
        return [gpu['usage'] for gpu in amd_gpus]  # Extract usage percentage for each GPU
    except Exception as e:
        print(f"Error getting AMD GPU usage: {e}")
        return []

def log_system_metrics(process):
    """Log CPU, memory, and GPU usage for the current process and its children."""
    while not stop_event.is_set():  # Check if the stop event is set
        # CPU usage
        cpu_usage = process.cpu_percent(interval=1)  # Total CPU usage including children
        
        # Memory usage
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / psutil.virtual_memory().total * 100  # RSS as a percentage of total memory

        # Get GPU usage if applicable
        gpu_usage = {}
        if NVIDIAGPU_IMPORTED:
            gpu_usage['NVIDIA'] = get_nvidia_usage()
        if AMDGPU_IMPORTED:
            gpu_usage['AMD'] = get_amd_usage()

        # Log the metrics to WandB
        wandb.log({
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_usage": gpu_usage  # Log GPU usage if available
        })

        # Optional: Print to console for real-time monitoring
        print(f"CPU Usage: {cpu_usage}, Memory Usage: {memory_usage}, GPU Usage: {gpu_usage}")
        time.sleep(1)  # Adjust the sleep time as needed

def start_logging_metrics():
    """Start logging system metrics in a separate thread."""
    process = psutil.Process()
    log_thread = threading.Thread(target=log_system_metrics, args=(process,), daemon=True)
    log_thread.start()
    return log_thread  # Return the thread for later use

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
    
    config = luigi.configuration.get_config()
    USE_WANDB = WANDB_IMPORTED

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
        for pipeline in results:

            if USE_WANDB:
                wandb.init(project="lot_sizing", name=str(extract_task_classes(str (pipeline))) + "_" + time.strftime("%Y%m%d-%H%M%S"))
                log_thread = start_logging_metrics()



            print("==============")
            print(deep_str(pipeline))
            print("\n")
            #luigi.build([r], local_scheduler=True, detailed_summary=True)
            print("\n")
            print("===============")

            if USE_WANDB:
                stop_event.set()  
                log_thread.join()
                wandb.finish()
                reset_stop_event()
    else:
        print("No results!")
