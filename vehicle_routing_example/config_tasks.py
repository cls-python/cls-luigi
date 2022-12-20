from os.path import join as pjoin
from os.path import dirname
from os import makedirs
import luigi
from cls_tasks import *
from configs import *



class AbstractNsBenchmark(CLSTask):
    abstract: bool = False
    config_resource_path = luigi.Parameter(default="")
    result_path = luigi.Parameter(default="")

    configs = ClsParameter(tpe={1: NSConfig1.return_type(), 2 : NSConfig2.return_type(), 3: NSConfig3.return_type()})
    config_domain = set(configs.tpe.keys())

    def requires(self):
       return self.configs(self.config_resource_path, self.result_path)

    def output(self):
       return self.input()

class ConfigLoaderHelper(CLSTask):
    abstract: bool = False
    load_task = ClsParameter(tpe={"ns" : AbstractNsBenchmark.return_type()})
    config_resource_path = luigi.Parameter(default="")
    result_path = luigi.Parameter(default="")
    given_config_domain = luigi.Parameter(default=set())

    config_domain = set(load_task.tpe.keys())

    def requires(self):
       return self.load_task(self.config_resource_path, self.result_path)

    def output(self):
        return self.input()

    def run(self):
        pass


# Sowas geht nicht
#
# class  MptopConfigLoader(CLSTask, globalConfig):
#     abstract = False
#     scoring_data = ClsParameter(tpe=AbstractScoringPhase.return_type())
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.config_domain = set()
#
#     def requires(self):
#         return self.scoring_data()
#
#     def output(self):
#         return {"mptop_config" : luigi.LocalTarget(pjoin(self.config_result_path,  self._get_variant_label() + "-" + "mptop_config.yaml"))}
#
#     def run(self):
#         with open(self.input()["scoring_method"].path, "r") as scoring_data:
#             scoring_type = scoring_data.readline()
#             self.config_domain.add(scoring_type)
#             other_target = yield ConfigLoaderHelper(self.global_config_path, self.config_result_path, self.config_domain)
#
#             # dynamic dependencies resolve into targets
#             f = other_target.open('r')
#             print("####################################")
#             print(f.readlines())
#
#
# class ConfigLoaderHelper(CLSTask):
#     abstract: bool = False
#     config_resource_path = luigi.Parameter(default="")
#     result_path = luigi.Parameter(default="")
#     given_config_domain = luigi.Parameter(default=set())
   
#     load_task = ClsParameter(default="", tpe={"ns" : AbstractNsBenchmark.return_type()})
#     config_domain = set(load_task.tpe.keys())

#     def requires(self):
#        return self.load_task(self.config_resource_path, self.result_path)

#     def output(self):
#         return self.input()

#     def run(self):
#         pass

