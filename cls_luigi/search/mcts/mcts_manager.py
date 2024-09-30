from __future__ import annotations
from typing import Dict, Any, List, Type, Optional, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import luigi
    import networkx as nx
    from cls_luigi.tools.constants import MAXIMIZE, MINIMIZE, MLMAXIMIZATIONMETRICS, MLMINIMIZATIONMETRICS
    from cls_luigi.search import (
        OnePlayerGame,
        SinglePlayerMCTS,
        SelectionPolicy,
        ExpansionPolicy,
        SimulationPolicy,
        ActionFilter, Node)
    from cls_luigi.search.core import TreeBase, Evaluator

import logging
from os.path import join as pjoin
from os import makedirs

from cls_luigi.search import (
    SelectionPolicy, UCT,
    RecursiveSinglePlayerMCTS, RandomExpansion,
    MCTSTreeWithGrammar, NodeFactory,
    HyperGraphGame, LuigiPipelineEvaluator)

from cls_luigi.tools.io_functions import dump_json

from cls_luigi.tools.constants import MINIMIZE, MAXIMIZE


class MCTSManager:
    """
    A class that manages the MCTS process.
    """

    def __init__(
        self,
        run_dir: Optional[str],
        pipeline_objects: List[luigi.Task],
        mcts_params: Dict[str, Any],
        hypergraph: nx.Graph,
        game_sense: Literal[MAXIMIZE, MINIMIZE],
        pipeline_metric: Literal[MLMAXIMIZATIONMETRICS, MLMINIMIZATIONMETRICS],

        evaluator_punishment_value: Union[int, float],

        game_cls: Type[OnePlayerGame] = HyperGraphGame,
        node_factory_cls: NodeFactory = NodeFactory,
        tree_cls: Type[TreeBase] = MCTSTreeWithGrammar,
        evaluator_cls: Type[Evaluator] = LuigiPipelineEvaluator,

        mcts_cls: Type[SinglePlayerMCTS] = RecursiveSinglePlayerMCTS,
        selection_policy: Type[SelectionPolicy] = UCT,
        expansion_policy: Type[ExpansionPolicy] = RandomExpansion,
        simulation_policy: Optional[Type[SimulationPolicy]] = None,
        pipeline_filters: Optional[List[ActionFilter]] = None,

        prog_widening_params: Optional[Dict[str, Any]] = None,
        component_timeout: Optional[int] = None,
        pipeline_timeout: Optional[int] = None,
        debugging_mode: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:

        self.run_dir = run_dir
        self.mcts_output_dir = pjoin(self.run_dir, "mcts") if self.run_dir else "mcts"
        self._make_run_dirs()
        self.pipeline_objects = pipeline_objects
        self.mcts_params = mcts_params
        self.hypergraph = hypergraph
        self.game_sense = game_sense
        self.pipeline_metric = pipeline_metric
        self.evaluator_punishment_value = evaluator_punishment_value
        self.game_cls = game_cls
        self.node_factory_cls = node_factory_cls
        self.tree_cls = tree_cls
        self.evaluator_cls = evaluator_cls
        self.mcts_cls = mcts_cls
        self.selection_policy = selection_policy
        self.expansion_policy = expansion_policy
        self.simulation_policy = simulation_policy
        self.pipeline_filters = pipeline_filters
        self.prog_widening_params = prog_widening_params
        self.component_timeout = component_timeout
        self.pipeline_timeout = pipeline_timeout
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.debugging_mode = debugging_mode

        self.evaluator = None
        self._init_evaluator()
        self.game = None
        self._init_game()
        self.mcts = None
        self._init_mcts()
        self._write_mcts_scenario()

    def _write_mcts_scenario(
        self,
        out_name: str = "mcts_scenario.json",
        out_path: Optional[str] = None
    ) -> None:

        scenario = {
            "type": self.mcts_cls.__name__,
            "mcts_parameters": self.mcts_params,
            "prog_widening_params": self.prog_widening_params,
            "policies": {
                "selection": self.selection_policy.__name__,
                "expansion": self.expansion_policy.__name__,
                "simulation": self.simulation_policy.__name__ if self.mcts_cls.__class__.__name__ == \
                                                                 "PureSinglePlayerMCTS" else None},
            "filters": [action_filter.__class__.__name__ for action_filter in
                        self.pipeline_filters] if self.pipeline_filters else [],
            "sense": "MINIMIZE" if self.game_sense == MINIMIZE else "MAXIMIZE" if self.game_sense == MAXIMIZE else "Undefined",
            "pipeline_metric": self.pipeline_metric,
            "component_timeout": self.component_timeout,
            "pipeline_timeout": self.pipeline_timeout,
        }

        if not out_path:
            out_path = self.mcts_output_dir

        dump_json(
            path=pjoin(out_path, out_name),
            obj=scenario)

    def _make_run_dirs(
        self
    ) -> None:

        makedirs(self.mcts_output_dir, exist_ok=False)

    def _init_evaluator(
        self
    ) -> None:
        self.evaluator = self.evaluator_cls(
            tasks=self.pipeline_objects,
            metric=self.pipeline_metric,
            task_timeout=self.component_timeout,
            pipeline_timeout=self.pipeline_timeout,
            punishment_value=self.evaluator_punishment_value,
            debugging_mode=self.debugging_mode,
            logger=self.logger
        )

    def _init_game(
        self
    ) -> None:
        self.game = self.game_cls(
            hypergraph=self.hypergraph,
            sense=self.game_sense,
            evaluator=self.evaluator,
            filters=self.pipeline_filters,
            logger=self.logger
        )

    def _init_mcts(
        self
    ) -> None:

        mcts_kwargs = {
            "game": self.game,
            "parameters": self.mcts_params,
            "selection_policy": self.selection_policy,
            "expansion_policy": self.expansion_policy,
            "tree_cls": self.tree_cls,
            "node_factory_cls": self.node_factory_cls,
            "prog_widening_params": self.prog_widening_params,
            "out_path": self.mcts_output_dir,
            "logger": self.logger
        }

        if self.mcts_cls.__class__.__name__ == "PureSinglePlayerMCTS":
            mcts_kwargs["simulation_policy"] = self.simulation_policy

        self.mcts = self.mcts_cls(**mcts_kwargs)

    def run_mcts(
        self
    ) -> Optional[List[Node]]:
        inc = self.mcts.run()
        if inc:
            return inc

    def save_results(self):
        self.mcts.save_results()

        eval_summary = self.evaluator.get_json_ready_summary()
        dump_json(
            path=pjoin(self.mcts_output_dir, "evaluator_summary.json"),
            obj=eval_summary
        )
