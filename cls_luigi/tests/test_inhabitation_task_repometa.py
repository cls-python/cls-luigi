# -*- coding: utf-8 -*-
#
# Apache Software License 2.0
#
# Copyright (c) 2022-2023, Jan Bessai, Anne Meyer, Hadi Kutabi, Daniel Scholtyssek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import unittest
import luigi

from os.path import dirname
from os import makedirs

from cls_luigi.inhabitation_task import ClsParameter, RepoMeta, LuigiCombinator
from cls_luigi.cls_tasks import ClsTask, ClsWrapperTask
from cls_luigi import RESULTS_PATH

class StartNode(ClsTask):
    abstract = False

    def output(self):
        return self.create_result_file(".dirs_created")

    def run(self):
        makedirs(dirname(RESULTS_PATH), exist_ok=True)
        with open(self.output().path, 'w') as file:
            pass


class SomeAbstractClass(ClsTask):
    abstract = True

    def requires(self):
        return {"start_node" : StartNode()}

    def output(self):
        return self.create_result_file( "-" + "some_abstract_class_result")

    def run(self):
        with open(self.output().path, 'w') as file:
            pass

class ConcreteClass4(SomeAbstractClass):
    abstract = False

class SomeAbstractAbstractClass(SomeAbstractClass):
    abstract = True

class ConcreteClass1(SomeAbstractAbstractClass):
    abstract = False

class ConcreteClass2(SomeAbstractAbstractClass):
    abstract = False

class ConcreteClass3(SomeAbstractAbstractClass):
    abstract = False

class SomeOtherAbstractAbstractClass(SomeAbstractClass):
    abstract = True

class ConcreteClassInAbstractChain(SomeOtherAbstractAbstractClass):
    abstract = False

class AbstractFromConcreteClassInChain(ConcreteClassInAbstractChain):
    abstract = True

class ConcreteClass5(AbstractFromConcreteClassInChain):
    abstract = False

class ConcreteClass6(AbstractFromConcreteClassInChain):
    abstract = False

class ConcreteClass7(AbstractFromConcreteClassInChain):
    abstract = False

class UnrelatedAbstractClass(ClsTask):
    abstract = True
    some_class = ClsParameter(tpe=SomeAbstractClass.return_type())

    def requires(self):
        return {"some_class" : self.some_class()}

    def output(self):
       return self.create_result_file( "-" + "unrelated_class_result")

    def run(self):
         with open(self.output().path, 'w') as file:
            pass

class UnrelatedConcreteClass1(UnrelatedAbstractClass):
    abstract = False

class UnrelatedConcreteClass2(UnrelatedAbstractClass):
    abstract = False

class EndNode(ClsTask):
    abstract = False
    unreleated_class = ClsParameter(tpe=UnrelatedAbstractClass.return_type())


    def requires(self):
        return {"unrelated_class" : self.unreleated_class()}

    def output(self):
        return self.create_result_file( "-" + "end_node_result")


    def run(self):
        with open(self.output().path, 'w') as file:
            pass

class EndEndNode(luigi.Task, LuigiCombinator):
    pass

class WrapperTask(ClsWrapperTask):
    abstract = False
    config = ClsParameter(tpe={1: ConcreteClass1.return_type(),
                                        "2": ConcreteClass2.return_type()})
    config_domain = {1, "2"}
    def requires(self):
        return self.config()

class TestRepositoryFilterMethods(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
         show_repository_and_subtypes_dict()

    def test_get_list_of_all_upstream_classes_ConcreteClass3(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_classes(ConcreteClass3),  [ConcreteClass3, SomeAbstractAbstractClass, SomeAbstractClass, ClsTask])

    def test_get_list_of_all_upstream_classes_ConcreteClass4(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_classes(ConcreteClass4),  [ConcreteClass4, SomeAbstractClass, ClsTask])

    def test_get_list_of_all_upstream_classes_ConcreteClass5(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_classes(ConcreteClass5),  [ConcreteClass5, AbstractFromConcreteClassInChain, ConcreteClassInAbstractChain, SomeOtherAbstractAbstractClass, SomeAbstractClass, ClsTask])

    def test_get_list_of_all_upstream_abstract_classes_ConcreteClass3(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_abstract_classes(ConcreteClass3), [SomeAbstractAbstractClass, SomeAbstractClass, ClsTask])

    def test_get_list_of_all_upstream_abstract_classes_ConcreteClass5(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_abstract_classes(ConcreteClass5), [AbstractFromConcreteClassInChain, SomeOtherAbstractAbstractClass, SomeAbstractClass, ClsTask])

    def test_get_list_of_all_upstream_abstract_classes_AbstractFromConcreteClassInChain(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_abstract_classes(AbstractFromConcreteClassInChain), [AbstractFromConcreteClassInChain, SomeOtherAbstractAbstractClass, SomeAbstractClass, ClsTask])

    def test_get_list_of_all_upstream_abstract_classes_ConcreteClassInAbstractChain(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_abstract_classes(ConcreteClassInAbstractChain), [SomeOtherAbstractAbstractClass, SomeAbstractClass, ClsTask])

    def test_get_all_upstream_classes_ConcreteClass1(self):
        self.assertTupleEqual(RepoMeta._get_all_upstream_classes(ConcreteClass1), (ConcreteClass1, [SomeAbstractAbstractClass, SomeAbstractClass, ClsTask]))

    def test_get_set_of_all_downstream_classes_ConcreteClassInAbstractChain(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_classes(ConcreteClassInAbstractChain), (ConcreteClassInAbstractChain, {AbstractFromConcreteClassInChain, ConcreteClass5, ConcreteClass6, ConcreteClass7}))

    def test_get_set_of_all_downstream_classes_SomeAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_classes(SomeAbstractClass), (SomeAbstractClass, {SomeAbstractAbstractClass, SomeOtherAbstractAbstractClass, ConcreteClass1, ConcreteClass2, ConcreteClass3, ConcreteClassInAbstractChain, AbstractFromConcreteClassInChain, ConcreteClass5, ConcreteClass6, ConcreteClass7, ConcreteClass4}))

    def test_get_all_downstream_abstract_classes_SomeAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_abstract_classes(SomeAbstractClass), (SomeAbstractClass, {SomeAbstractAbstractClass, SomeOtherAbstractAbstractClass, AbstractFromConcreteClassInChain}))

    def test_get_all_downstream_abstract_classes_UnrelatedAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_abstract_classes(UnrelatedAbstractClass), (UnrelatedAbstractClass, set()))

    def test_get_class_chain_SomeAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_class_chain(SomeAbstractClass), (SomeAbstractClass, [ClsTask], {SomeAbstractAbstractClass, ConcreteClass1, ConcreteClass2, ConcreteClass3, SomeOtherAbstractAbstractClass, ConcreteClassInAbstractChain, AbstractFromConcreteClassInChain, ConcreteClass5, ConcreteClass6, ConcreteClass7, ConcreteClass4}))

    def test_get_class_chain_ConcreteClass5(self):
        self.assertTupleEqual(RepoMeta._get_abstract_class_chain(ConcreteClass5), (ConcreteClass5, [AbstractFromConcreteClassInChain, ConcreteClassInAbstractChain, SomeOtherAbstractAbstractClass, SomeAbstractClass, ClsTask], set()))

    def test_get_abstract_class_chain_SomeAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_abstract_class_chain(SomeAbstractClass), (SomeAbstractClass, [ClsTask], {SomeAbstractAbstractClass, SomeOtherAbstractAbstractClass, AbstractFromConcreteClassInChain}))

    def test_get_maximal_shared_upper_classes_ConcreteClass5_ConcreteClass6(self):
        self.assertListEqual(RepoMeta._get_maximal_shared_upper_classes([ConcreteClass5, ConcreteClass6])[0], [ClsTask, SomeAbstractClass, SomeOtherAbstractAbstractClass, ConcreteClassInAbstractChain, AbstractFromConcreteClassInChain])

    def test_get_maximal_shared_upper_classes_ConcreteClass2_ConcreteClass7(self):
        self.assertListEqual(RepoMeta._get_maximal_shared_upper_classes([ConcreteClass2, ConcreteClass7])[0], [ClsTask, SomeAbstractClass])

    def test_get_all_downstream_concrete_classes_ConcreteClassInAbstractChain(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_concrete_classes(ConcreteClassInAbstractChain), (ConcreteClassInAbstractChain, {ConcreteClass5, ConcreteClass6, ConcreteClass7}))

    def test_get_all_downstream_concrete_classes_SomeAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_concrete_classes(SomeAbstractClass), (SomeAbstractClass, {ConcreteClass1, ConcreteClass2, ConcreteClass3, ConcreteClass4 ,ConcreteClass5, ConcreteClass6, ConcreteClass7, ConcreteClassInAbstractChain}))

    def test_delete_related_combinators_1(self):
        repository = RepoMeta.repository
        to_remove = []
        reference = repository.copy()
        for item in repository:
            if not isinstance(item, RepoMeta.ClassIndex):

                if issubclass(item.cls, (UnrelatedConcreteClass2, ConcreteClass4,ConcreteClass1, ConcreteClass2, ConcreteClass3)):
                    to_remove.append(item)

        for item in to_remove:
            reference.pop(item)

        self.assertDictEqual(reference, RepoMeta._delete_related_combinators([UnrelatedConcreteClass2, ConcreteClass4,ConcreteClass1, ConcreteClass2, ConcreteClass3]))

    def test_delete_related_combinators_2(self):
        repository = RepoMeta.repository
        to_remove = []
        reference = repository.copy()
        for item in repository:
            if not isinstance(item, RepoMeta.ClassIndex):

                if issubclass(item.cls, (ConcreteClass1, UnrelatedConcreteClass1)):
                    to_remove.append(item)

        for item in to_remove:
            reference.pop(item)

        self.assertDictEqual(reference, RepoMeta._delete_related_combinators([ConcreteClass1, UnrelatedConcreteClass1]))


    def test_filtered_repository_SomeAbstractAbstractClass(self):
        new_repo = RepoMeta._delete_related_combinators([ConcreteClass4, ConcreteClassInAbstractChain, ConcreteClass5, ConcreteClass6, ConcreteClass7])

        self.assertDictEqual(RepoMeta.filtered_repository([SomeAbstractAbstractClass]), new_repo)

    def test_filtered_repository_ConcreteClass1_ConcreteClass2(self):
        new_repo = RepoMeta._delete_related_combinators([ConcreteClass3])

        self.assertDictEqual(RepoMeta.filtered_repository([ConcreteClass1, ConcreteClass2]), new_repo)

    def test_filtered_repository_SomeOtherAbstractAbstractClass_UnrelatedConcreteClass1(self):
        repository = RepoMeta.repository
        to_remove = []
        new_repo = repository.copy()
        for item in repository:
            if not isinstance(item, RepoMeta.ClassIndex):

                if issubclass(item.cls, (UnrelatedConcreteClass2, ConcreteClass4,ConcreteClass1, ConcreteClass2, ConcreteClass3)):
                    to_remove.append(item)

        for item in to_remove:
            new_repo.pop(item)

        self.assertDictEqual(RepoMeta.filtered_repository([SomeOtherAbstractAbstractClass, UnrelatedConcreteClass1]), new_repo)


    def test_filtered_repository_with_deep_filter(self):
        repository = RepoMeta.repository
        to_remove = []
        new_repo = repository.copy()
        for item in repository:
            if not isinstance(item, RepoMeta.ClassIndex):

                if issubclass(item.cls, (ConcreteClass1, UnrelatedConcreteClass1, ConcreteClass5, ConcreteClass6, ConcreteClass7)):
                    to_remove.append(item)

        for item in to_remove:
            new_repo.pop(item)

        self.assertDictEqual(RepoMeta.filtered_repository([(SomeAbstractClass, [SomeAbstractAbstractClass, ConcreteClassInAbstractChain, ConcreteClass4]), (SomeAbstractAbstractClass, [ConcreteClass2, ConcreteClass3]), UnrelatedConcreteClass2]), new_repo)




def show_repository_and_subtypes_dict():

    print("Repo: ")
    repository = RepoMeta.repository
    for item in repository:
        print("#################")
        print("key: ", str(item), " :-> ", "value: ", str(repository[item]))
        print("#################")

    print("SubTypes:")
    subtypes = RepoMeta.subtypes
    for item in subtypes:
        print("*****************")
        print("key: ", str(item), " :-> ", "value: ", str(subtypes[item]))
        print("*****************")



if __name__=="__main__":
    unittest.main()
