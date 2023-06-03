.. image:: https://raw.githubusercontent.com/cls-python/cls-luigi/master/docs/images/cls-luigi-logo-transparent.png
  :target: https://github.com/cls-python/cls-luigi/
  :width: 60%
  :align: center

.. image:: https://img.shields.io/pypi/v/cls-luigi
        :target: https://pypi.python.org/pypi/cls-luigi

.. image:: https://img.shields.io/pypi/pyversions/cls-luigi
        :target: https://pypi.python.org/pypi/cls-luigi

.. image:: https://img.shields.io/pypi/l/cls-luigi?color=blue
        :target: https://github.com/cls-python/cls-luigi/blob/main/LICENSE

.. image:: https://img.shields.io/github/issues/cls-python/cls-luigi
        :target: https://github.com/cls-python/cls-luigi/issues

.. image:: https://github.com/cls-python/cls-python/actions/workflows/test-build-release.yaml/badge.svg
        :target: https://github.com/cls-python/cls-python/actions/workflows/test-build-release.yaml

.. image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Jekannadar/bc966a7d659af93f31be6b04415b9468/raw/covbadge.json
        :target: https://github.com/cls-python/cls-luigi/actions/workflows/test-build-release.yaml

.. image:: https://img.shields.io/badge/docs-online-green
        :target: https://cls-python.github.io/cls-python/readme.html
        :alt: Documentation Status
..
  .. image:: https://pyup.io/repos/github/cls-python/cls-luigi/shield.svg
     :target: https://pyup.io/repos/github/cls-python/cls-luigi/
     :alt: Updates

**************

TL:DR
-----

A framework for automated synthesis and execution of decision pipelines.

* Free software: Apache Software License 2.0
* Documentation: https://cls-python.github.io/cls-luigi/

What is it?
-----------

**In short:** our goal is to automatically create decision pipelines based on domain specific algorithmic repositories and depending on the available data!

In order to streamline decision-making processes, it is common practice to construct decision pipelines comprising various algorithms. These pipelines encompass tasks such as basic data pre-processing, statistical analysis, machine learning, and optimization. Consequently, building an effective pipeline necessitates expertise in the domains of machine learning, optimization, and the specific field at hand.

Moreover, it is important to note that there is no universally optimal pipeline for every problem in any domain. There is no single pipeline that represents the best choice, nor is there an ultimate model or configuration for machine learning approches. Likewise, the ideal model and solution method for optimization problems may vary. One of the main reasons for this can be the available data, which can change over time.

**CLS-Luigi** is an innovative pipeline tool designed to streamline the creation and execution of algorithmic pipelines by harnessing the power of combinatory logic.
It combines CLS-Python_, a type-theoretic framework for software component synthesis,  with Luigi_, a tool created by Spotify to build and execute pipelines for batch jobs. At its core, **CLS-Luigi** aims to integrate elements from *AutoML*, *Algorithm selection and configuration*, and *DevOps* to provide a comprehensive solution. However, it is important to note that **CLS-Luigi** is currently in its early stages of development, undergoing ongoing refinement and enhancement.


Main Features
-------------

Here are just a few of the things that **CLS-Luigi** does well:

- Allows natural modeling by specifying the necessary input to a component based on the types of the components it depends on.
- It's easy to define templates for algorithmic components and templates for pipeline structures.
- Good for batch-type pipelines where previous pipeline executions have no/little influence on the current run.
- Consistent creation of (all) pipeline variants based on a user defined repository of components, guaranteeing soundness and completeness by leveraging the CLS-Python_ framework.
- Efficient execution and resource optimization using features of Luigi_. Luigi optimizes resource usage through caching mechanisms, avoiding redundant computations by rerunning identical sub-pipelines only when necessary.
- The framework places significant importance on componentization and promotes a structured approach that considers the data flow, actions on the data, and data propagation to subsequent computational steps when implementing components. This enables the reuse of components in different domain-specific repositories and their smooth integration into diverse pipelines, as long as they align with the pipeline structure. Moreover, the framework offers flexibility to Python programmers, allowing them to expand existing domain-specific repositories by inheriting from a already existing component and implementing the necessary methods to create a new problem-specific or problem-agnostic component, thereby fostering customization and adaptability.
- Integrated visualizer that can display the repository and the scheduled pipelines.
- Python based framework: offers a extensive collection of specialized libraries and tools that offer pre-built algorithms, statistical functions, and visualization capabilities tailored for optimization, machine learning, and data analytics tasks. Moreover, Python's seamless integration with other technologies enhances its appeal and makes it a valuable tool for our projects.

Authors
-------

* Jan Bessai <jan.bessai@tu-dortmund.de>
* Anne Meyer <anne2.meyer@tu-dortmund.de>
* Hadi Kutabi <hadi.kutabi@tu-dortmund.de>
* Daniel Scholtyssek <daniel.scholtyssek@tu-dortmund.de>


.. _CLS-Python: https://github.com/cls-python/cls-python
.. _Luigi: https://github.com/spotify/luigi
