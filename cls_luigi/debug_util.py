from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import luigi

from luigi.task import flatten


def get_str_deps_tree(
    task: luigi.Task,
    indent: str = '',
    last: bool = True
):
    """
    Return a string representation of the tasks in dependency tree format.
    This is based on Luigi's tools.deps_tree.print_tree():
        https://luigi.readthedocs.io/en/stable/api/luigi.tools.deps_tree.html
    """

    name = task.__class__.__name__
    result = '\n' + indent
    if last:
        result += '└─--'
        indent += '    '
    else:
        result += '|---'
        indent += '|   '
    result += f'[{name}]'
    children = flatten(task.requires())
    for index, child in enumerate(children):
        result += get_str_deps_tree(child, indent, (index + 1) == len(children))
    return result
