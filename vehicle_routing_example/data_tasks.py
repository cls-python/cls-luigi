from os.path import join as pjoin
from os.path import dirname
from os import makedirs
import luigi
from cls_tasks import *


# self.resource_path = "resources/data"
# self.result_path = "results"

class LoadDataWrapper(CLSTask):
    abstract: bool = False
    instance_name = luigi.Parameter()
    load_revenue = luigi.BoolParameter(default=False)
    load_gold = luigi.BoolParameter(default=False)
    resource_path = luigi.Parameter(default="")
    result_path = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_dir_for_input_data = pjoin(str(self.result_path), "input_data/")

    def output(self):
        return {"customers" : luigi.LocalTarget(pjoin(self.result_dir_for_input_data,  self._get_variant_label() + "-" + "customers.csv")), "sales_person" : luigi.LocalTarget(pjoin(self.result_dir_for_input_data,  self._get_variant_label() + "-" + "sales_person.csv")), "customers_revenue" : luigi.LocalTarget(pjoin(self.result_dir_for_input_data,  self._get_variant_label() + "-" +"customers_revenue.csv")), "goldmember" : luigi.LocalTarget(pjoin(self.result_dir_for_input_data, self._get_variant_label() + "-" + "goldmember.csv")) }

    def run(self):
        makedirs(dirname(self.result_dir_for_input_data),exist_ok=True)
        with open(self.output()["customers"].path, "w") as customers_result:
            with open(pjoin(self.resource_path, "customers.csv"), "r") as target:
                customers_result.write(target.read())
        
        with open(self.output()["sales_person"].path, "w") as customers_result:
            with open(pjoin(self.resource_path, "sales_person.csv"), "r") as target:
                customers_result.write(target.read())
        
        if self.load_revenue:
            with open(self.output()["customers_revenue"].path, "w") as customers_result:
                with open(pjoin(self.resource_path, "customers_revenue.csv"), "r") as target:
                    customers_result.write(target.read())

        if self.load_gold:    
            with open(self.output()["goldmember"].path, "w") as customers_result:
                with open(pjoin(self.resource_path, "goldmember.csv"), "r") as target:
                    customers_result.write(target.read())
