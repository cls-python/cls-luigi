from typing import Dict, Set, List
import cls.fcl
from cls_luigi.grammar.helpers import remove_module_names  # returns a string with the module names removed


class ApplicativeTreeGrammarEncoder:
    def __init__(
        self,
        applicative_tree_grammar: Set[cls.fcl.Rule],
        start_symbol: str
    ) -> None:

        self.applicative_tree_grammar = applicative_tree_grammar
        self.start_symbol = start_symbol
        self.tree_grammar = {
            "start": self.start_symbol,
            "non_terminals": [],
            "terminals": [],
            "rules": {}
        }

    def encode_into_tree_grammar(
        self
    ) -> Dict[str, str | List[str] | Dict[str, List[str]]]:

        for rule in self.applicative_tree_grammar:
            if not isinstance(rule.target, cls.fcl.Arrow):
                target = remove_module_names(rule.target)
                if isinstance(rule, cls.fcl.Combinator):

                    combinator = remove_module_names(rule.combinator)
                    self._safe_update_rules(
                        target=target,
                        combinators=[combinator],
                        combinator_args=[])

                    self._safe_update_terms(non_terminals=[target], terminals=[combinator])

                elif isinstance(rule, cls.fcl.Apply):
                    combinators_and_args = self._compute_application(
                        function_type=rule.function_type,
                        argument_type=rule.argument_type)

                    for combinator, args in combinators_and_args.items():
                        self._safe_update_rules(
                            target=target,
                            combinators=[combinator],
                            combinator_args=args)

                        terminals = [combinator]
                        non_terminals = [target]
                        if args:
                            non_terminals.extend(args)

                        self._safe_update_terms(non_terminals=non_terminals, terminals=terminals)

        return self.tree_grammar

    def _compute_application(
        self,
        function_type: cls.fcl.Type,
        argument_type: cls.fcl.Type
    ) -> Dict[str, List[str]]:

        combinators_and_args = {}
        filtered_grammar = self._filter_applicative_tree_grammar(function_type)

        for rule in filtered_grammar:
            if isinstance(rule, cls.fcl.Combinator):
                c = remove_module_names(rule.combinator)
                if c not in combinators_and_args:
                    combinators_and_args[c] = []
                if remove_module_names(argument_type) not in combinators_and_args[c]:
                    combinators_and_args[c].append(remove_module_names(argument_type))

            elif isinstance(rule, cls.fcl.Apply):
                temp_combinators_and_args = self._compute_application(
                    function_type=rule.function_type,
                    argument_type=rule.argument_type
                )
                for k, v in temp_combinators_and_args.items():
                    if k not in combinators_and_args:
                        combinators_and_args[k] = []

                    combinators_and_args[k].extend(v)

                    if remove_module_names(argument_type) not in combinators_and_args[k]:
                        combinators_and_args[k].append(remove_module_names(argument_type))

        return combinators_and_args

    def _safe_update_rules(
        self,
        target: str,
        combinators: List[str | None],
        combinator_args: List[str | None],
        rules_key: str = "rules"
    ) -> None:

        if target not in self.tree_grammar[rules_key]:
            self.tree_grammar[rules_key][target] = {}

        for c in combinators:
            if c not in self.tree_grammar[rules_key][target]:
                self.tree_grammar[rules_key][target][c] = []

            if combinator_args not in self.tree_grammar[rules_key][target][c]:
                self.tree_grammar[rules_key][target][c].extend(combinator_args)

    def _filter_applicative_tree_grammar(
        self,
        function_type: cls.fcl.Type
    ) -> set[cls.fcl.Rule]:
        return set(filter(lambda rule: rule.target == function_type, self.applicative_tree_grammar))

    def _safe_update_terms(
        self,
        non_terminals: List[str],
        terminals: List[str]
    ) -> None:
        for n_terminal in non_terminals:
            if n_terminal not in self.tree_grammar["non_terminals"]:
                self.tree_grammar["non_terminals"].append(n_terminal)

        for terminal in terminals:
            if terminal not in self.tree_grammar["terminals"]:
                self.tree_grammar["terminals"].append(terminal)

