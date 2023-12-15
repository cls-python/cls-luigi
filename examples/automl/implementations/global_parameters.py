import luigi


class GlobalParameters(luigi.Config):
    """
    Global parameters for the pipeline, such as dataset id, name, seed, n_jobs, etc.
    """
    X_path = luigi.Parameter(default=None)
    y_path = luigi.Parameter(default=None)
    dataset_name = luigi.Parameter(default=None)
    n_jobs = luigi.IntParameter(default=1)
    seed = luigi.IntParameter(default=5)
