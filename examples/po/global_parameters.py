import luigi


class GlobalParameters(luigi.Config):

    dataset_name = luigi.Parameter(default="None")
    num_data = luigi.IntParameter(default=100)
    grid = luigi.TupleParameter(default=(5, 5))
    num_features = luigi.IntParameter(default=5)
    deg = luigi.IntParameter(default=1)
    noise_width = luigi.FloatParameter(default=0.0)
    seed = luigi.IntParameter(default=1)
    n_jobs = luigi.IntParameter(default=1)
