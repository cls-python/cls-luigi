import json
import os
import re
import threading
import time
from pathlib import Path


import luigi
import numpy as np
import pandas as pd
import psutil
from cls.debug_util import deep_str
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from lot_optimizers.groff_heuristic import GroffHeuristic
from lot_optimizers.least_unit_cost_method import LeastUnitCostMethod
from lot_optimizers.part_period_heuristic import PartPeriod
from lot_optimizers.silver_meal_heuristic import SilverMeal
from lot_optimizers.wagner_whitin import WagnerWhitin

from cls_luigi.inhabitation_task import ClsParameter, RepoMeta
from cls_luigi.utils.wandb import WandbTask, wandb_log


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
            self.log_artifact()
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
        return {"historic_demand": self.get_historic_demand()}

    def get_actual_demand(self):
        # just dummy values
        return [
            26,
            28,
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


# Create a global event to control the logging thread
stop_event = threading.Event()


def reset_stop_event():
    global stop_event
    stop_event = threading.Event()  # Create a new event, which is unset (False)


def log_system_metrics(process):
    """Log CPU and memory usage for the current process and its children."""
    while not stop_event.is_set():  # Check if the stop event is set
        # CPU usage
        cpu_usage = process.cpu_percent(
            interval=1
        )  # Total CPU usage including children

        # Memory usage
        memory_info = process.memory_info()
        memory_usage = (
            memory_info.rss / psutil.virtual_memory().total * 100
        )  # RSS as a percentage of total memory

        # Log the metrics to WandB
        wandb.log(
            {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage
            }
        )

        # Optional: Print to console for real-time monitoring
        print(
            f"CPU Usage: {cpu_usage}, Memory Usage: {memory_usage}"
        )
        time.sleep(1)  # Adjust the sleep time as needed


def start_logging_metrics():
    """Start logging system metrics in a separate thread."""
    process = psutil.Process()
    log_thread = threading.Thread(
        target=log_system_metrics, args=(process,), daemon=True
    )
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

            # wandb.init(project=PROJECT_NAME, name=PIPELINE_NAME)
            log_thread = start_logging_metrics()

            print("==============")
            print(deep_str(pipeline))
            print("\n")
            luigi.build([pipeline], local_scheduler=True, detailed_summary=True)
            print("\n")
            print("===============")

            
            stop_event.set()
            log_thread.join()
            # TODO finish wandb
            reset_stop_event()
    else:
        print("No results!")
