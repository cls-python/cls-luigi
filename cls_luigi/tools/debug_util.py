from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import luigi

from luigi.task import flatten


def get_str_deps_tree(
    task: luigi.Task,
    indent: str = '',
    last: bool = True,
    show_task_id: bool = False
):
    """
    Recursive function that return a string representation of the tasks in dependency tree format.
    This is based on luigi.tools.deps_tree.print_tree():
        https://luigi.readthedocs.io/en/stable/api/luigi.tools.deps_tree.html

    Parameters
    ----------
    task: luigi.Task
        The task to be added to the dependency tree.
    indent: str
        The current indentation level.
    last: bool
        Whether this task is the last in the list of dependencies.
    show_task_id: bool
        Whether to show the task id in the dependency tree instead of the name.
        (This can be useful for detecting duplicate tasks from the same type in the pipelines)

    Returns
    -------
    str
        The dependency tree representation of the tasks.
    """
    if show_task_id is True:
        name = task.task_id
    elif show_task_id is False:
        name = task.__class__.__name__

    deps_tree = '\n' + indent
    if last:
        deps_tree += '└─--'
        indent += '    '
    else:
        deps_tree += '|---'
        indent += '|   '
    deps_tree += f'[{name}]'
    children = flatten(task.requires())
    for index, child in enumerate(children):
        deps_tree += get_str_deps_tree(child, indent, (index + 1) == len(children), show_task_id)
    return deps_tree
