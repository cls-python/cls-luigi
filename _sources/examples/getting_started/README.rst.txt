Getting Started
===============

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

In Luigi, a task represents an atomic unit of work in a pipeline. Tasks are defined as Python classes that inherit from ``luigi.Task``. Each task should have an ``output()`` method that specifies the output it produces. The ``run()`` method contains the logic for executing the task. Luigi provides various other methods, such as ``requires()``, ``complete()``, and ``output()``, to define dependencies, check task completeness, and specify output targets.


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
problem of inhabitation. This problem denotes the question if a well-typed applicative term exists, that can be formed from a given repository Γ of typed combinators to satisfy a user-specified target type. The inhabitation problem can be expressed mathematically as follows: Γ ⊢? : σ. The repository Γ consists of combinators in the form (c : τ ) which can be read as ”In the respository Γ it is assumed that combinator c has type τ ”. CLS implements a inhabitation algorithm that uses the combinator types to determine which combinators can be applied to each other in order to satisfy the user-specified type σ. If a Term M exists, such as it satisfies the inhabitation request (Γ ⊢ M : σ), then we call M inhabitant of σ. If you want to learn more about the topic, you can read `this <https://eldorado.tu-dortmund.de/handle/2003/38387>`_.

In our use case CLS generates and executes Luigi pipelines. The Repository is composed of Classes that are derived from ``luigi.Task`` and ``cls_luigi.inhabitation_task.LuigiCombinator``. Finding the correct abstraction level, modeling components and creating the repository are sometimes the most difficult tasks when using CLS. It requires a lot of modeling experience as well as domain knowledge about the problem. To solve this challenge, CLS-Luigi uses Pythons reflection mechanisms to create the repository automatically. The user does not have to worry about this and simply implements components that he would have to implement anyways in some way.

CLS-Luigi generates all feasible luigi pipelines for a given target based on a repository of luigi-tasks using cls-python.

A Luigi tasks that not only inherit from ``luigi.Task`` but also from ``inhabitation_task.LuigiCombinator`` are automatically considered a component and thus will be added to the repository. All that needs to be done is to implement the methods ``run()``, ``output()`` and ``requires()``.


Basic concepts of CLS-Luigi
---------------------------

In the following section, we will give a short introduction on how to use CLS-Luigi. The examples found throughout can be found in the repository in the `examples/getting_started <https://github.com/cls-python/cls-luigi/tree/main/examples/getting_started/>`_ folder.

We will structure it as follows:

-  `How to run your very first CLS-Luigi Pipeline`_
-  `Define dependencies on other tasks <#dfo>`_
-  `Add variation points <#avp>`_
-  `Variation points as a dependency <#vpa>`_

How to run your very first CLS-Luigi Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


ML-Pipeline example
~~~~~~~~~~~~~~~~~~~

You can find the source for the example `here <https://github.com/cls-python/cls-luigi/tree/main/examples/getting_started/70_ml_example.py>`_:


Let’s consider an example where we predict the blood sugar level of some
patients. In this example we first start by loading the dataset from
Scikit-Learn, then we split it into 2 subsets for training and testing.

The first variation point is the scaling method. We introduce 2 concrete
implementation, namely ``RobustScaler``\ & ``MinMaxScaler``. After
scaling we have our second variation point which is the regression
model. Here we have also 2 concrete implementation, namely
``LinearRegression`` & ``LassoLars``.

Lastly we evaluate each regression model by predicting the testing
target and calculating the root mean squared error.

In this specific case we should have the following 4 pipelines:

We can see that ``RobustScaler`` & ``MinMaxScaler`` is required by both
the ``TrainLinearRegressionModel`` & ``TrainLassoLarsModel``, and
``EvaluateRegressionModel``

So the pipeline validation in this case looks like this:

.. code:: python

   validator = UniqueTaskPipelineValidator([FitTransformScaler])
   results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

Using dictionaries instead of lists in *requires* and *output* methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Code for this example is to be found
`here <hello_world_examples/_71_ML_example_with_dicts.py>`__

Till now, we only showed how to handle multiple dependencies and outputs
using lists. However, lists can be akward to handle, especially once you
deal with nested lists. Indexes are just not very easily readable.

In the `last
example <hello_world_examples/_70_ML_example_variation_point_multi_usage.py>`__
we returned the output of data splitting in a list as follows:

.. code:: python

       def output(self):
           return [
               luigi.LocalTarget("x_train.pkl"),
               luigi.LocalTarget("x_test.pkl"),
               luigi.LocalTarget("y_train.pkl"),
               luigi.LocalTarget("y_test.pkl")
           ]

       def run(self):
          ...
          ...
          ...
           X_train.to_pickle(self.output()[0].path)
           X_test.to_pickle(self.output()[1].path)
           y_train.to_pickle(self.output()[2].path)
           y_test.to_pickle(self.output()[3].path)

Note how we defined our outputs in *output* method, and had to feed in
the index number of corresponding LocalTarget in the *run* method. We
can do better than that! #### solution We can use dictionaries instead
of lists, and reference the corresponding LocalTargets using keys:

if your haven’t looked at the code for this example, you can also find
it `here <hello_world_examples/_71_ML_example_with_dicts.py>`__

.. code:: python

   class TrainTestSplit(luigi.Task, LuigiCombinator):
       abstract = False
       diabetes = inhabitation_task.ClsParameter(tpe=LoadDiabetesData.return_type())

       def output(self):
           return {
               "x_train": luigi.LocalTarget("x_train.pkl"),
               "x_test": luigi.LocalTarget("x_test.pkl"),
               "y_train": luigi.LocalTarget("y_train.pkl"),
               "y_test": luigi.LocalTarget("y_test.pkl"),
           }

       def run(self):
          ...
          ...
          ...

           X_train.to_pickle(self.output()["x_train"].path)
           X_test.to_pickle(self.output()["x_test"].path)
           y_train.to_pickle(self.output()["y_train"].path)
           y_test.to_pickle(self.output()["y_test"].path)

This is way easier to read! By using dictionaries we don’t need to
memorize the indexes of elements or even car about their order. We can
just use the keys without having to go back searching for the write
index.

Note: You can use this way in your *requires* method as well. and
instead of referencing as follows: \```python def requires(self): return
[self.my_dependency()]

def run(self): … # use the dependency output self.input()[0]

::


   your can just do the following instead:
   ```python
   def requires(self):
      return {"some_dependency": self.my_dependency()}

   def run(self):
      ...
      # use the dependency output
       self.input()["some_dependency"]

This obviously makes more sense with multiple dependencies.

Lastly: if you return dictionaries in both your *output* method and your
’requires\* method , don’t forget that you have to reference 2 keys now.
Still better as indexes :)

.. code:: python


   class SomeTaskA(luigi.Task, inhabitation_task.LuigiCombinator):
      def output(self):
         return {"output": Luigi.LocalTarget("output.suffix")}


   class SomeTaskA(luigi.Task, inhabitation_task.LuigiCombinator):
      dependency = ClsParameter(tpe= SomeTaskA.return_type())

      def requires(self):
         return {"output_from_SomeTaskA": self.dependency()}

      def run(self):
         ...
       # do something with your input (dependency)
           self.input()["output_from_SomeTaskA"]["output"].path



Known Issues
------------

1. Luigi on Windows got some `problems <https://luigi.readthedocs.io/en/stable/running_luigi.html?highlight=windows#luigi-on-windows>`_ due to the fact how windows is handling (or better not handling) forking of the python interpreter.
2. CLS-Python has the same problem since it is also using the multiprocessing package. This has an impact on the performance on windows, which makes the inhabitation process a bit longer than on linux.
3. The visualizer currently still has problems visualizing inheritance hierarchies and can currently only display pipelines that only contain Tasks that are direct implementations of luigi.Tasks and LuigiCombinator (see getting_started and lot_sizing examples). Therefore, we have currently refrained from using the classes in the module cls_tasks as base classes, as this would make the use of the visualizer impossible. On the other hand, only tasks are visualized, which are derived from LuigiCombinator and thus by using reflection are added to the Repository. However, pure Luigi tasks can also be used. While the visualizer's functionality is not yet fully comprehensive, it still offers valuable visualization capabilities for pipelines comprising tasks derived from luigi.Task and LuigiCombinator. Future enhancements and updates to the visualizer aim to address the challenges associated with visualizing inheritance hierarchies and expand the range of supported task types.
