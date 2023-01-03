import luigi

class MyConfig(luigi.Config):
    param = luigi.Parameter()

class MyConfigTask(luigi.Task):
    def run(self):
        # do something with MyConfig().param
        pass

    def output(self):
        return luigi.LocalTarget(MyConfig().param)

class MyDependentTask(luigi.Task):
    def requires(self):
        return MyConfigTask()

    def run(self):
        # do something with the output of MyConfigTask
        with self.input().open() as f:
            data = f.read()
        # do something with data
        pass

    def output(self):
        return luigi.LocalTarget('/path/to/output')


"""
In this example, MyConfig is a class that implements luigi.Config and has a parameter param. 
MyConfigTask is a task that does something with MyConfig().param. 
MyDependentTask depends on MyConfigTask and can use the output of MyConfigTask in its own run method.

To use this example, you will need to set the value of param in the configuration file. 
You can then run MyDependentTask by calling luigi.build([MyDependentTask()], local_scheduler=True).
"""
