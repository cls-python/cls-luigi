from typing import Union

import cls.fcl
import cls.types
from cls_luigi.inhabitation_task import RepoMeta


def remove_module_names(
    term: Union[cls.types.Type, object]
) -> str:
    if isinstance(term, cls.fcl.Arrow):
        return remove_module_name_from_arrow(term)
        # return remove_module_name_from_constructor_or_wrapped_task(term.path[-1])
    elif isinstance(term, cls.types.Constructor) or isinstance(term, RepoMeta.WrappedTask):
        return remove_module_name_from_constructor_or_wrapped_task(term)

    raise ValueError(f"Unsupported term type: {term}")


def remove_module_name_from_arrow(
    term: cls.fcl.Arrow
) -> str:
    string_arrow = str(term)
    split_arrow = string_arrow.split("->")
    cleaned_arrow = list(map(lambda target: target.split(".")[-1], split_arrow))

    return "-> ".join(cleaned_arrow)


def remove_module_name_from_constructor_or_wrapped_task(
    term: Union[cls.types.Constructor, RepoMeta.WrappedTask]
) -> str:
    return str(term).split(".")[-1]
