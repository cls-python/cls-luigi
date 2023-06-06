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
