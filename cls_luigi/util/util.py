import luigi

from cls_luigi import RESULTS_PATH


def get_unique_task_id(task: luigi.Task) -> str:
    """
    returns the unique task id of the given task.

    :param task: luigi.Task to get the id from
    :return: string unique task id
    """
    return task.task_id


def get_deps_tree_with_task_id(task: luigi.Task, indent: str = "", last: bool = True) -> str:
    """
    returns a string representation of the dependency tree of the given task with the task id of each task.

    :param task: luigi.Task to get the dependency tree from
    :return: string representation of the dependency tree for the given task with the task id of each task
    """

    task_id = get_unique_task_id(task)
    result = "\n" + indent
    if last is True:
        result += '└─--'
        indent += '   '
    else:
        result += '|--'
        indent += '|  '

    result += task_id
    children = luigi.task.flatten(task.requires())
    for index, child in enumerate(children):
        result += get_deps_tree_with_task_id(child, indent, index == len(children) - 1)
    return result


def save_deps_tree_with_task_id(task: luigi.Task, output_path: str = RESULTS_PATH, suffix: str = '.txt') -> None:
    """
    Generates and saves a string representation of the dependency tree of the given task with the task id of each task to the given output path.

    :param task: luigi.Task to get the dependency tree from
    :return: None
    """

    task_id = get_unique_task_id(task)
    full_output_path = output_path + f"/Dependency_tree_{task_id}{suffix}"
    with open(full_output_path, "w") as f:
        f.write(get_deps_tree_with_task_id(task))
