import luigi


class GlobalParameters(luigi.Config):
    """
    Global parameters for the pipeline, such as dataset id, name, seed, n_jobs, etc.
    """
    dataset_id = luigi.IntParameter(default=None)
    n_jobs = luigi.IntParameter(default=1)
    seed = luigi.IntParameter(default=5)
