Getting Started
===============

**The Getting Started section is still a work in progress, and certain concepts have not been fully explained yet. We are in the process of adding them and they will be included shortly. As a result, please keep this in mind while reading this section.**

The purpose of this guide is to illustrate some of the main features that CLS-Luigi provides. It assumes a very basic knowledge of python. Please refer to our :doc:`installation instructions <../../installation>` for installing CLS-Luigi.

.. _luigi_getting_started:

Getting Started with pure Luigi
-------------------------------

As CLS-Luigi leverages the capabilities provided by `Luigi <https://luigi.readthedocs.io/en/stable/index.html>`__ , it becomes essential to provide a brief introduction to Luigi itself, ensuring a solid foundation for understanding CLS-Luigi's functionalities.


Introduction to Basic Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Luigi is a powerful open-source framework developed by Spotify for building data pipelines. It allows you to define and execute complex workflows with multiple tasks and dependencies. Luigi provides a clean and simple interface for managing dependencies and orchestrating the execution of tasks in a distributed environment.
Luigi models pipelines as Directed Acyclic Graphs (DAGs), where tasks are represented as nodes, and dependencies between tasks are represented as edges. This graph structure allows you to define complex workflows with interdependent tasks. Luigi ensures that tasks are executed in the correct order based on their dependencies, minimizing redundant computations.

What is a Luigi Task?
~~~~~~~~~~~~~~~~~~~~~

In Luigi, a task represents an atomic unit of work in a pipeline. Tasks are defined as Python classes that inherit from ``luigi.Task``. Each task should have an ``output()`` method that specifies the output it produces, a ``requires()`` that specifies what other tasks are required to be run beforehand and a ``run()`` method that contains the logic for executing the task. Luigi provides various other methods, for more information check out the `Luigi documentation <https://luigi.readthedocs.io/en/stable/api/luigi.task.html?highlight=task>`__.
When you implement your own tasks, you often override these base class methods.


Advantage of Luigi
~~~~~~~~~~~~~~~~~~

One of the key advantages of Luigi is its built-in caching mechanism. Luigi automatically tracks the completeness of tasks and their dependencies. If a task has already been completed and its dependencies have not changed, Luigi uses the cached result, avoiding unnecessary re-computation. This caching mechanism significantly improves the efficiency of pipeline execution, especially in scenarios where tasks have expensive computations or external dependencies.

Example
~~~~~~~

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

This introductory information should be sufficient to understand CLS-Luigi. There are of course many more possibilities that Luigi offers, e.g. different wrapper tasks to interact with different data and databases. More information can be found in the `documentation of Luigi <https://luigi.readthedocs.io/en/stable/>`_.

From Luigi to CLS-Luigi
-----------------------

CLS-Luigi combines Luigi with the `(CL)S Framework <https://eldorado.tu-dortmund.de/handle/2003/38387>`__. CLS is a Type-Theoretic Framework for Software Component Synthesis.

CLS fundamentally solves the type-theoretical
problem of inhabitation. This problem denotes the question if a well-typed applicative term exists, that can be formed from a given repository Γ of typed combinators to satisfy a user-specified target type. The inhabitation problem can be expressed mathematically as follows: Γ ⊢? : σ. The repository Γ consists of combinators in the form (c : τ) which can be read as ”In the respository Γ it is assumed that combinator c has type τ ”. CLS implements a inhabitation algorithm that uses the combinator types to determine which combinators can be applied to each other in order to satisfy the user-specified type σ. If a Term M exists, such as it satisfies the inhabitation request (Γ ⊢ M : σ), then we call M inhabitant of σ. If you want to learn more about the topic, you can read `this <https://eldorado.tu-dortmund.de/handle/2003/38387>`_.

In our use case CLS generates and executes Luigi pipelines. The Repository is composed of Classes that are derived from ``luigi.Task`` and ``cls_luigi.inhabitation_task.LuigiCombinator``. Finding the correct abstraction level, modeling components and creating the repository are sometimes the most difficult tasks when using CLS. It requires a lot of modeling experience as well as domain knowledge about the problem. To solve this challenge, CLS-Luigi uses Pythons reflection mechanisms to create the repository automatically. The user does not have to worry about this and simply implements components that he would have to implement anyways in some way.

CLS-Luigi generates all feasible luigi pipelines for a given target based on a repository of luigi-tasks using cls-python.

A Luigi tasks that not only inherit from ``luigi.Task`` but also from ``inhabitation_task.LuigiCombinator`` are automatically considered a component and thus will be added to the repository. All that needs to be done is to implement the methods ``run()``, ``output()`` and ``requires()``.


Workflow of CLS-Luigi
---------------------

To create a pipeline, we need to follow these steps:

Step 1: Designing the Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we need to determine the structure of our pipeline. Consider the work steps (tasks) involved, their inputs and outputs, and how the data should flow through the pipeline. This careful planning will lay the foundation for a successful implementation.
At this stage, it is essential to consider the potential variation points within our pipeline. These points represent locations where algorithmic components can be swapped out. While these components typically serve the same purpose, they may employ different methodologies, such as alternative heuristics for a given problem. By identifying these variation points, we can create abstract classes that can be used to specify method implementations that must be utilized in the child classes. For instance, this could be used to employ the template method design pattern to enforce a required algorithmic flow within the components.

Step 2: Implementing the Work Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once we have designed our pipeline, it's time to implement the work steps. We achieve this by creating classes that inherit from `luigi.Task`. These classes will represent the individual tasks within our pipeline. To create a new task class, we override the necessary methods (`requires()`, `run()`, `output()`) as explained in the documentation on `What is a Luigi Task?`_.

While implementing the new task class, it's essential to ensure that it also inherits from `LuigiCombinator`. This step enables us to add these classes to the component repository through reflection, making them eligible for pipeline synthesis and variance consideration.

Ensure that each task is appropriately labeled as either an abstract or a concrete class. This can be accomplished by modifying the class variable *abstract* and setting it to either *True* or *False*.

Step 3: Writing the Boilerplate Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the final step, we only require a small portion of boilerplate code to initiate the synthesis and subsequent execution of our pipelines. This boilerplate code remains mostly consistent across different implementations. In the future, we aim to integrate it into our framework so that users will not have to write it themselves, unless they specifically wish to make modifications for a particular use case.

By following these steps, we can fully leverage the capabilities of CLS-Luigi to construct intricate and efficient pipelines for various data processing tasks.

Feel free to refer to the CLS-Luigi documentation for more comprehensive information on the framework's features and advanced usage.

Now, let's delve into each step and thoroughly explore the detailed capabilities of CLS-Luigi by examining concrete examples.


Exploring CLS-Luigi with an Example
-----------------------------------

In the following section, we will give a short introduction on how to use CLS-Luigi. You can locate the source code for the example in the repository's `examples/getting_started <https://github.com/cls-python/cls-luigi/tree/main/examples/getting_started/>`_ folder.
We will gradually incorporate additions or modifications in a step-by-step manner. As a result, the repository will encompass all iterations of the process.

*coming soon*

Peculiarities of CLS-Luigi
~~~~~~~~~~~~~~~~~~~~~~~~~~

*coming soon*

How to visualize your Pipeline?
-------------------------------

**The current version of the Vizualizer faces challenges when visualizing pipelines that are structured using abstract chains. Further details regarding this matter can be accessed in the Known Issues section. We are already working on an improved version**

*coming soon*

Known Issues
------------

1. Luigi on Windows got some `problems <https://luigi.readthedocs.io/en/stable/running_luigi.html?highlight=windows#luigi-on-windows>`_ due to the fact how windows is handling (or better not handling) forking of the python interpreter.
2. CLS-Python has the same problem since it is also using the multiprocessing package. This has an impact on the performance on windows, which makes the inhabitation process a bit longer than on linux.
3. The visualizer currently still has problems visualizing inheritance hierarchies and can currently only display pipelines that only contain Tasks that are direct implementations of luigi.Tasks and LuigiCombinator (see getting_started and lot_sizing examples). Currently it looks like the visualizer can't handle abstract chains (a class marked as abstract that inherits from another class that is marked abstract). Therefore, we have currently refrained from using the classes in the module cls_tasks as base classes, as this would make the use of the visualizer impossible. On the other hand, only tasks are visualized, which are derived from LuigiCombinator and thus by using reflection are added to the Repository. However, pure Luigi tasks can also be used. While the visualizer's functionality is not yet fully comprehensive, it still offers valuable visualization capabilities for pipelines comprising tasks derived from luigi.Task and LuigiCombinator. Future enhancements and updates to the visualizer aim to address the challenges associated with visualizing inheritance hierarchies and expand the range of supported task types.
