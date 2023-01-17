# hash_name_example

Here we want to see if we can implement a way to save the _get_variant_label result (and thus the taken way through the modeled pipeline variants) and save it to disc. The filename should be the hash, the first line in the file should be the taken way.

## Problem with multiple workers and global/class variables

When using multiple workers in Luigi, tasks are executed independently and in parallel. Each worker runs a separate instance of a task, so if a task writes to a global or class variable, it will only affect the instance of that task that is running on that specific worker. Therefore, it is not recommended to use global or class variables to store the results of tasks when using multiple workers.

A better approach would be to have the task write its results to an external storage such as a database or a file. That way, all the results can be collected and analyzed in a single location, regardless of which worker ran the task.

Alternatively, you can use a shared memory storage system like Redis, where each worker writes the result into the shared memory storage, then a final task can read the results from the storage and aggregate them.

It is also worth noting that if you are using global variable it could lead to race condition and unexpected behavior, specially if multiple worker are writing to the same variable.

