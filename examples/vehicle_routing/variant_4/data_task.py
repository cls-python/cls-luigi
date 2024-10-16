from os.path import join as pjoin
import luigi
from cls_tasks import *


class LoadDataWrapper(CLSTask):
    abstract: bool = False
    instance_name = luigi.Parameter()
    load_revenue = luigi.BoolParameter(default=False)
    load_gold = luigi.BoolParameter(default=False)
    resource_path = luigi.Parameter(default="")
    result_path = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def output(self):
        return {"customers": luigi.LocalTarget(pjoin(self.result_path,  self._get_variant_label() + "-" + "customers.csv")), "sales_person": luigi.LocalTarget(pjoin(self.result_path,  self._get_variant_label() + "-" + "sales_person.csv")), "customers_revenue": luigi.LocalTarget(pjoin(self.result_path,  self._get_variant_label() + "-" + "customers_revenue.csv")), "goldmember": luigi.LocalTarget(pjoin(self.result_path, self._get_variant_label() + "-" + "goldmember.csv"))}

    def run(self):
        with open(self.output()["customers"].path, "w") as customers_result:
            with open(pjoin(self.resource_path, "customers.csv"), "r") as target:
                customers_result.write(target.read())

        with open(self.output()["sales_person"].path, "w") as customers_result:
            with open(pjoin(self.resource_path, "sales_person.csv"), "r") as target:
                customers_result.write(target.read())

            with open(self.output()["customers_revenue"].path, "w") as customers_result:
                with open(pjoin(self.resource_path, "customers_revenue.csv"), "r") as target:
                    if self.load_revenue:
                        customers_result.write(target.read())
                    else:
                        pass

            with open(self.output()["goldmember"].path, "w") as customers_result:
                with open(pjoin(self.resource_path, "goldmember.csv"), "r") as target:
                    if self.load_gold:
                        customers_result.write(target.read())
                    else:
                        pass
