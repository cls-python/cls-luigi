Getting Started
===============

The purpose of this guide is to illustrate some of the main features that **cls-luigi** provides. It assumes a very basic knowledge of python. Please refer to our :doc:`installation instructions <../../installation>` for installing **cls-luigi**.

.. _luigi_getting_started:

Getting Started with pure Luigi
-------------------------------

As **cls-luigi** leverages the capabilities provided by Luigi, it becomes essential to provide a brief introduction to Luigi itself, ensuring a solid foundation for understanding cls-luigi's functionalities.


Introduction to Basic Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Luigi is a powerful open-source framework developed by Spotify for building data pipelines. It allows you to define and execute complex workflows with multiple tasks and dependencies. Luigi provides a clean and simple interface for managing dependencies and orchestrating the execution of tasks in a distributed environment.
Luigi models pipelines as Directed Acyclic Graphs (DAGs), where tasks are represented as nodes, and dependencies between tasks are represented as edges. This graph structure allows you to define complex workflows with interdependent tasks. Luigi ensures that tasks are executed in the correct order based on their dependencies, minimizing redundant computations.

What is a Luigi Task?
~~~~~~~~~~~~~~~~~~~~~

In Luigi, a task represents an atomic unit of work in a pipeline. Tasks are defined as Python classes that inherit from ``luigi.Task``. Each task should have an ``output()`` method that specifies the output it produces. The ``run()`` method contains the logic for executing the task. Luigi provides various other methods, such as ``requires()``, ``complete()``, and ``output()``, to define dependencies, check task completeness, and specify output targets.


Advantage of Luigi
~~~~~~~~~~~~~~~~~~

One of the key advantages of Luigi is its built-in caching mechanism. Luigi automatically tracks the completeness of tasks and their dependencies. If a task has already been completed and its dependencies have not changed, Luigi uses the cached result, avoiding unnecessary re-computation. This caching mechanism significantly improves the efficiency of pipeline execution, especially in scenarios where tasks have expensive computations or external dependencies.

Small Example
~~~~~~~~~~~~~

To illustrate the concepts, here's a simple example using Luigi. You can find the example `here <https://github.com/cls-python/cls-luigi/tree/main/examples/getting_started/10_getting_started_luigi.py>`_:


.. code-block:: python
    :linenos:
    :caption: Example: 10_getting_started_luigi.py

    import luigi

    class TaskA(luigi.Task):
        def requires(self):
            return None

        def output(self):
            return luigi.LocalTarget("output/taskA_output.txt")

        def run(self):
            with self.output().open('w') as f:
                f.write("Task A completed")

    class TaskB(luigi.Task):
        def requires(self):
            return TaskA()

        def output(self):
            return luigi.LocalTarget("output/taskB_output.txt")

        def run(self):
            with self.input().open() as input_file, self.output().open('w') as output_file:
                data = input_file.read()
                output_file.write("Task B completed with input: " + data)

    if __name__ == '__main__':
        luigi.build([TaskB()], local_scheduler=True)

In this example, we have two tasks: ``TaskA`` and ``TaskB``. ``TaskB`` requires ``TaskA`` as its dependency. ``TaskA`` writes a simple message to its output file, and ``TaskB`` reads the output of ``TaskA`` and appends a message before writing it to its own output file.

To run the example, execute the Python script:

.. code-block:: bash

    $ python 10_getting_started_luigi.py

The execution of ``TaskB`` will automatically trigger the execution of ``TaskA`` due to the defined dependency. The output files for each task will be created in the specified ``output/`` directory.

With this introductory information, you should have it easier to understand the functionalities of cls-luigi.

From Luigi to CLS-Luigi
-----------------------

CLS-Luigi combines the pipelining tool
`Luigi <https://luigi.readthedocs.io/en/stable/index.html>`__ with the
`(CL)S Framework <https://eldorado.tu-dortmund.de/handle/2003/38387>`__. CLS is a Type-Theoretic Framework for Software Component Synthesis.

CLS fundamentally solves the type-theoretical
problem of inhabitation. This problem denotes the question if a well-typed applicative term exists,
that can be formed from a given repository Γ of typed combinators to satisfy a user-specified target
type. The inhabitation problem can be expressed mathematically as follows: Γ ⊢? : σ. The repository
Γ consists of combinators in the form (c : τ ) which can be read as ”In the respository Γ it is assumed
that combinator c has type τ ”. CLS implements a inhabitation algorithm that uses the combinator
types to determine which combinators can be applied to each other in order to satisfy the user-specified
type σ. If a Term M exists, such as it satisfies the inhabitation request (Γ ⊢ M : σ), then we
call M inhabitant of σ. If you want to learn more about the topic, you can read `this <https://eldorado.tu-dortmund.de/handle/2003/38387>`_.

In our use case CLS generates and executes Luigi pipelines. The Repository is composed of Classes that are derived from ``luigi.Task`` and ``cls_luigi.inhabitation_task.LuigiCombinator``. Finding the correct abstraction level, modeling components and creating the repository are sometimes the most difficult tasks when using CLS. It requires a lot of modeling experience as well as domain knowledge about the problem. To solve this challenge, CLS-Luigi uses Pythons reflection mechanisms to create the repository automatically. The user does not have to worry about this and simply implement components.

# add info about target type

cls-luigi generates all feasible luigi pipelines for a given target based on a repository of luigi-tasks using cls-python.

A luigi tasks that not only inherit from ``luigi.Task`` but also from our LuigiCombinator ``inhabitation_task.LuigiCombinator`` are automatically considered as luigi tasks that are part of the task repository. All we need to implement is a pure luigi task, i.e., the methods ``run()``, ``output()`` and ``requires()``.

In the following we give a short intro of how to use cls-luigi. For each section, we also provide you with a running example in the folder


Known Issues
------------

1. Luigi on Windows got some `problems <https://luigi.readthedocs.io/en/stable/running_luigi.html?highlight=windows#luigi-on-windows>`_ due to the fact how windows is handling (or better not handling) forking of the python interpreter.
2. CLS-Python has the same problem since it is also using the multiprocessing package. This has an impact on the performance on windows, which makes the inhabitation process a bit longer than on linux.
3. The visualizer currently still has problems visualizing inheritance hierarchies and can currently only display pipelines that only contain Tasks that are direct implementations of luigi.Tasks and LuigiCombinator (see getting_started and lot_sizing examples). Therefore, we have currently refrained from using the classes in the module cls_tasks as base classes, as this would make the use of the visualizer impossible. On the other hand, only tasks are visualized, which are derived from LuigiCombinator and thus by using reflection are added to the Repository. However, pure Luigi tasks can also be used. While the visualizer's functionality is not yet fully comprehensive, it still offers valuable visualization capabilities for pipelines comprising tasks derived from luigi.Task and LuigiCombinator. Future enhancements and updates to the visualizer aim to address the challenges associated with visualizing inheritance hierarchies and expand the range of supported task types.
