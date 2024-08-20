import logging

import luigi
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes

from cls_luigi.grammar import ApplicativeTreeGrammarEncoder
from cls_luigi.grammar.hypergraph import plot_hypergraph_components, get_hypergraph_dict_from_tree_grammar, \
    build_hypergraph
from cls_luigi.inhabitation_task import ClsParameter, LuigiCombinator, RepoMeta
from cls_luigi.search.mcts.game import HyperGraphGame
from cls_luigi.search.mcts.policy import UCT
from cls_luigi.search.mcts.sp_mcts import SP_MCTS
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator


class Input(luigi.Task, LuigiCombinator):
    abstract = True


class csv(Input):
    abstract = False


class DataPrep(luigi.Task, LuigiCombinator):
    abstract = True


class NumPrep(DataPrep):
    abstract = True


class Imputer(NumPrep):
    abstract = True
    original_features_and_target = ClsParameter(tpe=Input.return_type())

    def requires(self):
        return [self.original_features_and_target]


class s_imp(Imputer):
    abstract = False


class Scaler(NumPrep):
    abstract = True
    imputed_features = ClsParameter(tpe=Imputer.return_type())

    def requires(self):
        return [self.imputed_features]


class minmax(Scaler):
    abstract = False


class robust(Scaler):
    abstract = False


# class standard(Scaler):
#     abstract = False
#
#
# class power(Scaler):
#     abstract = False
#
#
# class quantile(Scaler):
#     abstract = False


class FeatPrep(DataPrep):
    abstract = True
    numerically_processed_features = ClsParameter(tpe=NumPrep.return_type())
    original_features_and_target = ClsParameter(tpe=Input.return_type())

    def requires(self):
        return [self.numerically_processed_features, self.original_features_and_target]


class pca(FeatPrep):
    abstract = False


class ica(FeatPrep):
    abstract = False


# class feature_agglomeration(FeatPrep):
#     abstract = False
#
#
# class kernel_pca(FeatPrep):
#     abstract = False
#
#
# class nystroem(FeatPrep):
#     abstract = False
#
#
# class rbfsampler(FeatPrep):
#     abstract = False
#
#
# class polynomial(FeatPrep):
#     abstract = False
#
#
# class random_trees_embedding(FeatPrep):
#     abstract = False
#
#
# class select_from_extra_trees(FeatPrep):
#     abstract = False
#
#
# class select_from_svc(FeatPrep):
#     abstract = False
#
#
# class select_percentile(FeatPrep):
#     abstract = False
#
#
# class select_rates(FeatPrep):
#     abstract = False


class CLF(luigi.Task, LuigiCombinator):
    abstract = True
    processed_features = ClsParameter(tpe=DataPrep.return_type())
    original_features_and_targets = ClsParameter(tpe=Input.return_type())

    def requires(self):
        return [self.processed_features, self.original_features_and_targets]


class random_forest(CLF):
    abstract = False


class adaboost(CLF):
    abstract = False


# class bernoulli_nb(CLF):
#     abstract = False
#
#
# class decision_tree(CLF):
#     abstract = False
#
#
# class gradient_boosting(CLF):
#     abstract = False
#
#
# class gaussian_nb(CLF):
#     abstract = False
#
#
# class ida(CLF):
#     abstract = False
#
#
# class linear_svc(CLF):
#     abstract = False
#
#
# class multinomial_nb(CLF):
#     abstract = False
#
#
# class passive_aggressive(CLF):
#     abstract = False
#
#
# class mlp(CLF):
#     abstract = False
#
#
# class svc(CLF):
#     abstract = False
#
#
# class sgd(CLF):
#     abstract = False
#
#
# class qda(CLF):
#     abstract = False
#
#
# class knn(CLF):
#     abstract = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    target_class = CLF

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

    validator = UniqueTaskPipelineValidator([Scaler])

    results = [t() for t in inhabitation_result.evaluated[0:max_results]]

    tree_grammar = ApplicativeTreeGrammarEncoder(rules, CLF.__name__).encode_into_tree_grammar()

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
    plot_hypergraph_components(hypergraph, "binary_clf.png", start_node="CLF", node_size=5000, node_font_size=11)

    print(tree_grammar)

    params = {
        "num_iterations": 200,
        "exploration_param": 0.5,
        "num_simulations": 1,
    }
    game = HyperGraphGame(hypergraph)

    mcts = SP_MCTS(
        game=game,
        parameters=params,
        selection_policy=UCT,
    )
    mcts.run()
    mcts.draw_tree("nx_di_graph.png", plot=True)
    mcts.shut_down("mcts.pkl", "nx_di_graph.pkl")
