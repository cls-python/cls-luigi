# Variant 1

This variant implements the pipeline shown in pipeline_to_implement.svg.
There were some difficulties implementing the connection between the chosen scoring method and
which config to load.
In my masters thesis, having full access to CLS and how to repository is structured, it was more easy to keep track of decisions
CLS made and use that knowledge to decide which config to load.  

So in this variant all config files were simply varied without connection to the scoring method.

## Problems while implementing:

- unable to parametrize Tasks that contain luigi Parameter and ClsParameters.
  - For that reason hard to use luigi.config in general?!
- Pipeline Visualization does not work for my implementation because of keyerror in an internal data structure that is build during the Pipeline Visualization Step for the static pipeline (will create an example).
- Right now there is now easy way to interact with the automatically generated repository to filter out wanted pipelines. CLS Luigi just always generates every possible pipeline.
- Implemented a Aggregation Task that searches the "global" best pipeline (pipeline that gets you the best ObjVal out of the mptop solver).
  - the implementation works and gets you the best pipeline, but i have a feeling that it is only "luck" because in theory does scheduled tasks could run in parallel trying to read/write the same file. So far there was no corruption or race condition ...
  - for that i tried to use luigis build in batch modus, where you can run identical tasks that only differentiate in for example a given luigi.dataparameter in a batch, where only the newest task actually will be executed. But since i was unable to work with Cls and luigi Parameters combined, i couldn't get it to work. Also since CLS Luigi spilts off into different tasks if a required task is an abstract task, this should not work in general?!
