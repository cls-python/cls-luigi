# Variant 3

This variant implements the pipeline shown in pipeline_to_implement.svg.

Everything is the same as in variant_1. This variation was used to implement a repository filter method
to be able to specify which implemented tasks (concrete or abstract) should be used to create variations with CLS Luigi.
With does new methods one should be able to implement a pipeline configuration tool, where you are able to easily chose
which task (and thus ultimately which combinators) should be used to form valide solutions for the inhabitation problem at hand.

To be able to easily test and implement the new method, there was a test pipeline implemented which can be seen in the
pipeline_filter_repository_example.svg. This pipeline is used during the tests, which can be found in tests/test_inhabitation_task.py.

## potential downsides/problems

- since we are very open with the formate of how one could implement a pipeline (abstract tasks that inherits from concrete tasks, tasks sharing upstream base classes ...), it is very hard to automatically mark combinators to keep or delete.
  - to solve that in a understandable way, if not specified, we are only going up one abstract level and not further.
  - also we filter out global_shared_upstream_classes that are shared between all given tasks.
