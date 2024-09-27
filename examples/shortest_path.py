import logging

import luigi

from cls_luigi.grammar import MetaPipelineEnumerator, sort_topologically
from cls_luigi.grammar.hypergraph import get_hypergraph_dict_from_tree_grammar, render_hypergraph_components, \
    build_hypergraph
from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes

from cls_luigi.grammar.encoding import ApplicativeTreeGrammarEncoder
from cls_luigi.search.mcts.game import HyperGraphGame
from cls_luigi.search.mcts.policy import UCT
from cls_luigi.search.mcts.pure_mcts import PureSinglePlayerMCTS
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator


class Input(luigi.Task, LuigiCombinator):
    abstract = True


class pyepo_data(Input):
    abstract = False


class OptSol(luigi.Task, LuigiCombinator):
    abstract = True
    split_dataset = ClsParameter(tpe=Input.return_type())

    def requires(self):
        return [self.split_dataset]


class GurobiOpt(OptSol):
    abstract = False


class Sol(luigi.Task, LuigiCombinator):
    abstract = True
    split_dataset = ClsParameter(tpe=Input.return_type())

    def requires(self):
        return [self.split_dataset]


class TwpStage(Sol):
    abstract = True


class Reg(TwpStage):
    abstract = True


class rf(Reg):
    abstract = False


# class lr(Reg):
#     abstract = False
#
#
# class gbm(Reg):
#     abstract = False


class EndToEnd(Sol):
    abstract = True


class SPOPlus(EndToEnd):
    abstract = False
    sols_and_objs = ClsParameter(tpe=OptSol.return_type())

    regressor = None
    optimizer = None
    train_loader = None
    test_loader = None

    test_predictions = None

    def requires(self):
        return [self.sols_and_objs, self.split_dataset]


class PredBasedSol(luigi.Task, LuigiCombinator):
    abstract = True
    predictions = ClsParameter(tpe=Sol.return_type())

    def requires(self):
        return [self.predictions]


class GurobiPredSol(PredBasedSol):
    abstract = False


class Evaluation(luigi.Task, LuigiCombinator):
    abstract = True
    optimal_sols_and_objs = ClsParameter(tpe=OptSol.return_type())
    predicted_solution = ClsParameter(tpe=PredBasedSol.return_type())
    predictions = ClsParameter(tpe=Sol.return_type())
    split_dataset = ClsParameter(tpe=Input.return_type())

    def requires(self):
        return [self.optimal_sols_and_objs, self.predicted_solution, self.predictions, self.split_dataset]


class Regret(Evaluation):
    abstract = False


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    target_class = Evaluation

    target = target_class.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)

    inhabitation_result = fcl.inhabit(target)
    rules = inhabitation_result.rules

    print("Enumerating results...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual

    results = [t() for t in inhabitation_result.evaluated[0:max_results]]

    tree_grammar = ApplicativeTreeGrammarEncoder(rules, target_class.__name__).encode_into_tree_grammar()

    print("start", tree_grammar["start"])
    print("non_terminals", tree_grammar["non_terminals"])
    print("terminals", tree_grammar["terminals"])
    print("rules")
    for k, v in tree_grammar["rules"].items():
        print(k)
        print(v)
        print()

    print(tree_grammar)

    hypergraph_dict = get_hypergraph_dict_from_tree_grammar(tree_grammar)
    hypergraph = build_hypergraph(hypergraph_dict)
    render_hypergraph_components(hypergraph, "binary_clf.png", start_node="CLF", node_size=5000, node_font_size=11)

    print(tree_grammar)

    params = {
        "num_iterations": 200,
        "exploration_param": 0.5,
        "num_simulations": 1,
    }
    game = HyperGraphGame(hypergraph)

    mcts = PureSinglePlayerMCTS(
        game=game,
        parameters=params,
        selection_policy=UCT,
    )
    mcts.run()
    mcts.draw_tree("nx_di_graph.png", plot=True)
    mcts.shut_down("mcts.pkl", "nx_di_graph.pkl")
