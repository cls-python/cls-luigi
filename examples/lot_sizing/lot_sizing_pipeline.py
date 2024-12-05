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

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class WandbTask(luigi.Task, LuigiCombinator):
    """Base class for tasks that support wandb tracking"""
    enable_wandb = luigi.BoolParameter(default=False, description="Enable Weights & Biases tracking")
    wandb_project = luigi.Parameter(default="lot-sizing-optimization", description="Weights & Biases project name")
    wandb_entity = luigi.Parameter(default=None, description="Weights & Biases entity (username or team name)")
    wandb_tags = luigi.ListParameter(default=[], description="Tags for the wandb run")

    def log_metrics(self, metrics, commit=True):
        """Log metrics to wandb if enabled"""
        if self.enable_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, commit=commit)

    def log_config(self, config):
        """Log config to wandb if enabled"""
        if self.enable_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.config.update(config)

    def log_artifact(self, artifact_path, name, type):
        """Log an artifact to wandb if enabled"""
        if self.enable_wandb and WANDB_AVAILABLE and wandb.run is not None:
            artifact = wandb.Artifact(name=name, type=type)
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)

    def log_hyperparameters(self, params):
        """Log hyperparameters to wandb if enabled"""
        if self.enable_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.config.update(params)

    def log_system_metrics(self):
        """Log system metrics to wandb"""
        if self.enable_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "execution_time": self.execution_time  # Track this in your run method
            })

    def log_metrics_and_plots(self, orders, metrics, demand):
        """Log metrics and plots to wandb"""
        if self.enable_wandb:
            self.log_metrics(metrics)
            
            # Create visualizations
            self.plot_line_series(
                xs=[[i for i in range(len(orders))]], 
                ys=[orders],
                keys=["Order Quantities"],
                title="Order Quantities over Time",
                xname="Period"
            )
            
            inventory_levels = self.calculate_inventory_levels(orders, demand)
            self.plot_line_series(
                xs=[[i for i in range(len(inventory_levels))]], 
                ys=[inventory_levels],
                keys=["Inventory Level"],
                title="Inventory Levels over Time",
                xname="Period"
            )

    def log_if_enabled(self, log_function, *args, **kwargs):
        """Generalized logging method that checks if wandb is enabled."""
        if self.enable_wandb:
            log_function(*args, **kwargs)

    def track_experiment(self, orders, metrics, demand, cost):
        """Handles logging and tracking for wandb."""
        self.log_if_enabled(self.log_metrics, metrics)
        self.log_if_enabled(self.log_hyperparameters, {
            "fixed_cost": cost["fixedCost"],
            "variable_cost": cost["varCost"],
            "prediction_horizon": self.prediction_horizon,
            "optimizer": self.__class__.__name__,
            "total_demand": sum(demand),
            "avg_demand": np.mean(demand),
            "std_demand": np.std(demand),
            "max_demand": max(demand),
            "min_demand": min(demand),
            "demand_length": len(demand)
        })
        self.log_if_enabled(self.log_system_metrics)
        self.log_if_enabled(self.log_metrics_and_plots, orders, metrics, demand)

    def run(self):
        """Override this method in derived classes to implement task logic"""
        raise NotImplementedError()

class InitializeWandb(WandbTask):
    """Task to initialize wandb. Other tasks can depend on this to ensure wandb is initialized."""
    
    def output(self):
        return luigi.LocalTarget('data/wandb_initialized.txt')

    def run(self):
        if self.enable_wandb and WANDB_AVAILABLE:
            if not wandb.run:
                wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    tags=self.wandb_tags
                )
            
            # Save initialization info
            Path('data').mkdir(exist_ok=True)
            with self.output().open('w') as f:
                json.dump({
                    'project': self.wandb_project,
                    'entity': self.wandb_entity,
                    'tags': self.wandb_tags,
                    'run_id': wandb.run.id if wandb.run else None
                }, f)
        else:
            # Create marker file even if wandb is disabled
            Path('data').mkdir(exist_ok=True)
            with self.output().open('w') as f:
                json.dump({'wandb_enabled': False}, f)

class GetCost(WandbTask):
    abstract = False

    def requires(self):
        if self.enable_wandb:
            return InitializeWandb(enable_wandb=self.enable_wandb,
                                 wandb_project=self.wandb_project,
                                 wandb_entity=self.wandb_entity,
                                 wandb_tags=self.wandb_tags)

    def output(self):
        return luigi.LocalTarget('data/cost.json')

    def run(self):
        d = {
            "fixedCost": 400,  # Bestellkosten
            "varCost": 1,  # Lagerhaltungssatz
        }
        os.makedirs('data', exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(d, f, indent=4)
        
        if self.enable_wandb:
            self.log_config(d)
            self.log_artifact(self.output().path, "cost_parameters", "parameters")

class GetHistoricDemand(WandbTask):
    def output(self):
        print("GetHistoricDemand: output")
        return luigi.LocalTarget('data/historic_demand.csv')

    def run(self):
        print("====== GetHistoricDemand: run")
        with self.output().open('w') as f:
            f.write("1, 5, 7, 8, 9, 10, 14, 16, 19, 21, 19, 23, 24, 26, 26, "
                    "26, 28, 26, 28, 30")

class PredictDemand(WandbTask):
    abstract = True
    get_historic_demand = ClsParameter(tpe=GetHistoricDemand.return_type())
    prediction_horizon = 8

    def requires(self):
        reqs = {"historic_demand": self.get_historic_demand()}
        if self.enable_wandb:
            reqs["wandb"] = InitializeWandb(enable_wandb=self.enable_wandb,
                                          wandb_project=self.wandb_project,
                                          wandb_entity=self.wandb_entity,
                                          wandb_tags=self.wandb_tags)
        return reqs

class PredictDemandByAverage(PredictDemand):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/predicted_demand_by_average.pkl')

    def run(self):
        print("============= PredictDemandByAverage: run")
        with self.input()["historic_demand"].open() as infile:
            text = infile.read()
            l = [int(t) for t in text.split(",")]
            avg = int(sum(l) / len(l) + 0.5)
            predicted = [avg for i in range(self.prediction_horizon)]
            data = {'predicted_demand': predicted}
            df_predicted = pd.DataFrame(data)

            # Log demand prediction metrics
            if self.enable_wandb:
                self.log_metrics({
                    "prediction_method": "average",
                    "mean_predicted_demand": np.mean(predicted),
                    "std_predicted_demand": np.std(predicted),
                    "min_predicted_demand": min(predicted),
                    "max_predicted_demand": max(predicted)
                })

                # Create demand prediction plot
                self.plot_line_series(
                    xs=[[i for i in range(len(predicted))]], 
                    ys=[predicted],
                    keys=["Predicted Demand"],
                    title="Predicted Demand over Time",
                    xname="Period"
                )

            df_predicted.to_pickle(self.output().path)

class PredictDemandByLinearRegression(PredictDemand):
    abstract = False

    def output(self):
        return luigi.LocalTarget(
            'data/predicted_demand_by_linear_regression.pkl')

    def run(self):
        print("============= PredictDemandByLinearRegression: run")
        with self.input()["historic_demand"].open() as infile:
            print("I'm just a mock for Linear Regression!!!")
            predicted = [10 + i for i in range(self.prediction_horizon)]
            data = {'predicted_demand': predicted}
            df_predicted = pd.DataFrame(data)

            # Log demand prediction metrics
            if self.enable_wandb:
                self.log_metrics({
                    "prediction_method": "linear_regression",
                    "mean_predicted_demand": np.mean(predicted),
                    "std_predicted_demand": np.std(predicted),
                    "min_predicted_demand": min(predicted),
                    "max_predicted_demand": max(predicted)
                })

                # Create demand prediction plot
                self.plot_line_series(
                    xs=[[i for i in range(len(predicted))]], 
                    ys=[predicted],
                    keys=["Predicted Demand"],
                    title="Predicted Demand over Time",
                    xname="Period"
                )

            df_predicted.to_pickle(self.output().path)

class OptimizeLots(WandbTask):
    abstract = True
    predicted_demand = ClsParameter(tpe=PredictDemand.return_type())
    get_cost = ClsParameter(tpe=GetCost.return_type())

    def requires(self):
        reqs = {
            "cost": self.get_cost(),
            "demand": self.predicted_demand()
        }
        if self.enable_wandb:
            reqs["wandb"] = InitializeWandb(enable_wandb=self.enable_wandb,
                                          wandb_project=self.wandb_project,
                                          wandb_entity=self.wandb_entity,
                                          wandb_tags=self.wandb_tags)
        return reqs

    def run(self):
        print(f"============= {self.__class__.__name__}: run")
        with self.input()["cost"].open() as cost_file:
            cost = json.load(cost_file)

        with self.input()["demand"].open('rb') as demand_file:
            df_demand = pd.read_pickle(demand_file)
            demand = df_demand['predicted_demand'].tolist()

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
        return luigi.LocalTarget('data/' + self._get_variant_label() + "-" + 'optimize_lots_by_groff.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByGroff: run")
        optimizer = GroffHeuristic()
        orders = optimizer.run(cost, demand)
        
        metrics = {"total_cost": sum(orders), "fixed_costs": 0, "variable_costs": sum(orders)}
        if self.enable_wandb:
            metrics["criterion_value"] = (2 * cost["fixedCost"]) / cost["varCost"]
        
        return orders, metrics

class OptimizeLotsByWagnerWhitin(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/' + self._get_variant_label() + "-" + 'optimize_lots_by_wagner_within.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByWagnerWithin: run")
        optimizer = WagnerWhitin()
        orders = optimizer.run(cost, demand)
        
        metrics = {"total_cost": sum(orders), "fixed_costs": 0, "variable_costs": sum(orders)}
        if self.enable_wandb:
            metrics.update({
                "algorithm": "wagner_whitin",
                "is_optimal": True
            })
        
        return orders, metrics

class OptimizeLotsBySilverMeal(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/' + self._get_variant_label() + "-" + 'optimize_lots_by_silver_meal.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsBySilverMeal: run")
        optimizer = SilverMeal()
        orders = optimizer.run(cost, demand)
        
        metrics = {"total_cost": sum(orders), "fixed_costs": 0, "variable_costs": sum(orders)}
        if self.enable_wandb:
            # Calculate period costs only if wandb is enabled
            inventory = 0
            period_costs = []
            for i, (order, dem) in enumerate(zip(orders, demand)):
                inventory += order - dem
                if inventory > 0:
                    period_costs.append(inventory * cost["varCost"])
            
            metrics.update({
                "avg_period_cost": np.mean(period_costs) if period_costs else 0,
                "max_period_cost": max(period_costs) if period_costs else 0
            })
        
        return orders, metrics

class OptimizeLotsByLeastUnitCost(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/' + self._get_variant_label() + "-" + 'optimize_lots_by_least_unit_cost.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByLeastUnitCost: run")
        optimizer = LeastUnitCostMethod()
        orders = optimizer.run(cost, demand)
        
        metrics = {"total_cost": sum(orders), "fixed_costs": 0, "variable_costs": sum(orders)}
        if self.enable_wandb:
            total_units = sum(demand)
            metrics.update({
                "cost_per_unit": metrics["total_cost"] / total_units if total_units > 0 else 0,
                "total_units": total_units
            })
        
        return orders, metrics

class OptimizeLotsByPartPeriod(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/' + self._get_variant_label() + "-" + 'optimize_lots_by_part_period.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByPartPeriod: run")
        optimizer = PartPeriod()
        orders = optimizer.run(cost, demand)
        
        metrics = {"total_cost": sum(orders), "fixed_costs": 0, "variable_costs": sum(orders)}
        if self.enable_wandb:
            # Calculate part periods only if wandb is enabled
            inventory = 0
            part_periods = []
            for i, (order, dem) in enumerate(zip(orders, demand)):
                inventory += order - dem
                if inventory > 0:
                    part_periods.append(inventory * cost["varCost"])
            
            metrics.update({
                "avg_part_period": np.mean(part_periods) if part_periods else 0,
                "max_part_period": max(part_periods) if part_periods else 0,
                "num_part_periods": len(part_periods)
            })
        
        return orders, metrics

if __name__ == "__main__":
    from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
    from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = OptimizeLots.return_type()
    repository = RepoMeta.repository
    StaticJSONRepo(RepoMeta).dump_static_repo_json()
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
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")
