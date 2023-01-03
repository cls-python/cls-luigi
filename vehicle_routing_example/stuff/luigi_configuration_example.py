import luigi
import luigi.configuration

class MyTask(luigi.Task):
    def run(self):
        # access a value in the 'mysection' section of the configuration file
        param = luigi.configuration.get_config().get('mysection', 'myparam')
        # do something with param
        pass

    def output(self):
        return luigi.LocalTarget('/path/to/output')

if __name__ == '__main__':
    # load the configuration file before running the task
    luigi.configuration.LuigiConfigParser.add_config_path('/path/to/my_config.cfg')
    luigi.build([MyTask()], local_scheduler=True)

"""
In this example, MyTask is a task that accesses a value in the mysection section of the 
configuration file using luigi.configuration.get_config().get('mysection', 'myparam'). 
The configuration file is loaded using luigi.configuration.LuigiConfigParser.add_config_path() before the task is run.

You can also use the --local-scheduler flag when running Luigi from the command line to specify 
the path to the configuration file:

luigi --local-scheduler --config-path /path/to/my_config.cfg MyTask
"""
