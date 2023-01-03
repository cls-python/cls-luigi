import luigi

class MyWrapperTask(luigi.WrapperTask):
    def requires(self):
        # yield a list of tasks that the wrapper task depends on
        yield MyTask1()
        yield MyTask2()
        yield MyTask3()

    def run(self):
        # do something with the outputs of the yielded tasks
        with self.input()[0].open() as f1:
            data1 = f1.read()
        with self.input()[1].open() as f2:
            data2 = f2.read()
        with self.input()[2].open() as f3:
            data3 = f3.read()
        # combine the outputs of the tasks
        output = data1 + data2 + data3
        # write the combined output to the output file
        with self.output().open('w') as f:
            f.write(output)

    def output(self):
        return luigi.LocalTarget('/path/to/output')
x
