import luigi


class GlobalParameters(luigi.Config):
    seed = luigi.IntParameter(default=1)
    n_jobs = luigi.IntParameter(default=1)
