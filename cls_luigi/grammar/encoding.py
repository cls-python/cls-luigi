from typing import Iterable, Any
import cls.fcl
from cls_luigi.grammar.helpers import remove_module_names  # returns a string with the module names removed


class ApplicativeTreeGrammarEncoder:
    def __init__(
        self,
        applicative_tree_grammar: Iterable[cls.fcl.Rule]
    ) -> None:

        self.applicative_tree_grammar = applicative_tree_grammar
        self.tree_grammar = {}

    def encode_into_tree_grammar(
        self
    ) -> dict[str | Any: dict[str | Any: set[str | Any]]]:

        for rule in self.applicative_tree_grammar:
            target = remove_module_names(rule.target)
            if isinstance(rule, cls.fcl.Combinator):

                combinator = remove_module_names(rule.combinator)
                self._safe_update_tree_grammar_dict(
                    target=target,
                    combinators={combinator},
                    combinator_args=set())

            elif isinstance(rule, cls.fcl.Apply):
                combinators, args = self._compute_rules(
                    function_type=rule.function_type,
                    argument_type=rule.argument_type)

                self._safe_update_tree_grammar_dict(
                    target=target,
                    combinators=combinators,
                    combinator_args=args)

        return self.tree_grammar

    def _compute_rules(
        self,
        function_type: cls.fcl.Type,
        argument_type: cls.fcl.Type
    ) -> tuple[set[str | Any], set[str | Any]]:

        combinators = set()
        args = set()
        filtered_grammar = self._filter_applicative_tree_grammar(function_type)

        for rule in filtered_grammar:
            if isinstance(rule, cls.fcl.Combinator):
                combinators.add(remove_module_names(rule.combinator))
                args.add(remove_module_names(argument_type))
            elif isinstance(rule, cls.fcl.Apply):
                temp_combinators, temp_args = self._compute_rules(
                    function_type=rule.function_type,
                    argument_type=rule.argument_type
                )
                combinators.update(temp_combinators)
                args.update(temp_args)
                args.add(remove_module_names(argument_type))

        return combinators, args

    def _safe_update_tree_grammar_dict(
        self,
        target: str | Any,
        combinators: set[str | Any],
        combinator_args: set[str | Any]
    ) -> None:

        if target not in self.tree_grammar:
            self.tree_grammar[target] = {}

        for c in combinators:
            if c not in self.tree_grammar[target]:
                self.tree_grammar[target][c] = set()
            self.tree_grammar[target][c].update(combinator_args)

    def _filter_applicative_tree_grammar(
        self,
        function_type: cls.fcl.Type
    ) -> set[cls.fcl.Rule]:
        return set(filter(lambda rule: rule.target == function_type, self.applicative_tree_grammar))
