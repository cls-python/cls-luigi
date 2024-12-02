"""
Task analysis utilities for cls_luigi.

This module provides functionality for analyzing and sorting Luigi tasks based on their
complexity (number of tasks) and other characteristics.
"""

from typing import Any, List, TypeVar, Callable
from cls_luigi.inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta
import luigi
import inspect

T = TypeVar('T')

# Cache for storing subclasses to avoid repeated lookups
_subclass_cache = {}

def _get_subclasses(cls: type) -> List[type]:
    """Get non-abstract LuigiCombinator subclasses with caching."""
    if cls not in _subclass_cache:
        _subclass_cache[cls] = [
            subcls for subcls in cls.__subclasses__()
            if issubclass(subcls, LuigiCombinator) and not subcls.abstract
        ]
    return _subclass_cache[cls]

def _inspect_generator_tasks(task: Any, visited: set) -> set:
    """
    Helper function to recursively inspect generator tasks and their requirements.
    Returns a set of all tasks found in the generator's frame without executing any task code.
    """
    found_tasks = set()
    
    try:
        # Instead of calling run(), inspect the code object directly
        if hasattr(task, 'run'):
            run_method = task.run
            if hasattr(run_method, '__code__'):
                code = run_method.__code__
                # Look for task classes in code names
                for name in code.co_names:
                    if name.endswith('Task'):
                        # Look in the module globals first
                        module = inspect.getmodule(task)
                        if module and hasattr(module, name):
                            task_class = getattr(module, name)
                            if isinstance(task_class, type) and issubclass(task_class, luigi.Task):
                                try:
                                    task_instance = task_class()
                                    found_tasks.add(task_instance)
                                    # Recursively check if this task also has generator run method
                                    if _is_generator_task(task_instance):
                                        found_tasks.update(_inspect_generator_tasks(task_instance, visited))
                                except Exception:
                                    pass
                                    
                # Also look in closure variables
                if hasattr(run_method, '__closure__') and run_method.__closure__:
                    for cell in run_method.__closure__:
                        if isinstance(cell.cell_contents, luigi.Task):
                            found_tasks.add(cell.cell_contents)
                            if _is_generator_task(cell.cell_contents):
                                found_tasks.update(_inspect_generator_tasks(cell.cell_contents, visited))
    except Exception:
        pass
        
    return found_tasks

def _is_generator_task(task: Any) -> bool:
    """Check if a task has a generator run method."""
    return (hasattr(task, 'run') and 
            hasattr(task.run, '__code__') and 
            task.run.__code__.co_flags & 0x20)  # CO_GENERATOR

def _get_task_from_parameter(param: Any, visited: set) -> set:
    """Helper function to extract tasks from parameters."""
    found_tasks = set()
    
    if isinstance(param, luigi.Task):
        if id(param) not in visited:
            found_tasks.add(param)
    elif isinstance(param, dict):
        for value in param.values():
            found_tasks.update(_get_task_from_parameter(value, visited))
    elif isinstance(param, (list, tuple)):
        for item in param:
            found_tasks.update(_get_task_from_parameter(item, visited))
    elif hasattr(param, 'tpe'):
        # Handle ClsParameter
        param_type = param.tpe
        impl_types = param_type.values() if isinstance(param_type, dict) else [param_type]
        for impl_type in impl_types:
            if isinstance(impl_type, type) and issubclass(impl_type, luigi.Task):
                try:
                    task_instance = impl_type()
                    if id(task_instance) not in visited:
                        found_tasks.add(task_instance)
                except Exception:
                    pass
    
    return found_tasks

def _get_tasks_from_requires(task: Any, visited: set) -> set:
    """Helper function to recursively get tasks from requires() method."""
    found_tasks = set()
    
    try:
        reqs = task._requires() if hasattr(task, '_requires') else task.requires()
        if isinstance(reqs, luigi.Task):
            if id(reqs) not in visited:
                found_tasks.add(reqs)
        elif isinstance(reqs, (list, tuple)):
            for req in reqs:
                found_tasks.update(_get_task_from_parameter(req, visited))
        elif isinstance(reqs, dict):
            for req in reqs.values():
                found_tasks.update(_get_task_from_parameter(req, visited))
    except Exception:
        pass
        
    return found_tasks

def count_tasks(task: Any, visited: set = None) -> int:
    """
    Count the total number of tasks in a task tree, including requirements and implementations.
    
    This function analyzes a task and counts:
    1. The task itself
    2. All required tasks (from requires() and _requires())
    3. All ClsParameter fields and their implementations
    4. Tasks with different config indices
    5. Dynamically yielded tasks from run()
    
    Args:
        task: A Luigi task instance
        visited: Set of visited task instances to prevent cycles
        
    Returns:
        int: The total number of tasks in the task tree
    """
    if task is None or not isinstance(task, luigi.Task):
        return 0
        
    if visited is None:
        visited = set()
        
    task_id = id(task)
    if task_id in visited:
        return 0
        
    visited.add(task_id)
    total = 1  # Count current task
    
    # Add config index variations if task is a LuigiCombinator
    if isinstance(task, LuigiCombinator):
        config_domain = getattr(task, 'config_domain', None)
        if config_domain:
            total += len(config_domain) - 1  # -1 because we already counted the base task
    
    # Get tasks from parameters
    if hasattr(task, 'get_params'):
        for param_name, param_obj in task.get_params():
            param_value = getattr(task, param_name, None)
            found_tasks = _get_task_from_parameter(param_value, visited)
            if isinstance(param_obj, ClsParameter):
                found_tasks.update(_get_task_from_parameter(param_obj, visited))
            for found_task in found_tasks:
                total += count_tasks(found_task, visited)
    
    # Get tasks from requires()
    for req_task in _get_tasks_from_requires(task, visited):
        total += count_tasks(req_task, visited)
    
    # Get tasks from run() generators
    if _is_generator_task(task):
        for yielded_task in _inspect_generator_tasks(task, visited):
            if id(yielded_task) not in visited:
                visited.add(id(yielded_task))
                total += count_tasks(yielded_task, visited) + 1  # +1 to count the yielded task itself
    
    return total

def sort_tasks_by_complexity(tasks: List[T], reverse: bool = False) -> List[T]:
    """
    Sort tasks by their complexity (number of tasks in their tree).
    
    Args:
        tasks: List of tasks to sort
        reverse: If True, sort from most complex to least complex
                If False, sort from least complex to most complex
                
    Returns:
        List[T]: Sorted list of tasks
    """
    try:
        return sorted(tasks, key=lambda x: (count_tasks(x), str(x)), reverse=reverse)
    except Exception:
        return tasks

def create_task_sorter(key_func: Callable[[T], Any] = None, reverse: bool = False) -> Callable[[List[T]], List[T]]:
    """
    Create a task sorting function with custom key and order.
    
    Args:
        key_func: Optional function to extract a secondary sort key
                 If None, uses str(task) as secondary key
        reverse: If True, sort from most complex to least complex
                If False, sort from least complex to most complex
                
    Returns:
        Callable[[List[T]], List[T]]: A function that sorts tasks by complexity
    """
    if key_func is None:
        key_func = str
        
    def sorter(tasks: List[T]) -> List[T]:
        return sorted(tasks, key=lambda x: (count_tasks(x), key_func(x)), reverse=reverse)
        
    return sorter
