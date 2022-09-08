import json
from os.path import join

from luigi import WrapperTask

from inhabitation_task import RepoMeta


class StaticJSONRepo:
    """
    Creates a json representation of all components in repository, their types,
    their concrete implementations (if the task is abstract), and their dependencies.


    usage:
      static_repo = StaticJSONRepo(RepoMeta)      static_repo.dump_repo(where to dump your json)
    """

    def __init__(self, repo_meta):
        self.repo_meta = repo_meta
        self.repository = repo_meta.repository
        self.concrete_to_abstract_mapper = {}  # used for mapping the concrete implementations to their abstract components
        self.repo_dict = {}

        self._construct_repo_dict()

    @staticmethod
    def _prettify_name(name):
        return name.split(".")[-1]

    def _construct_repo_dict(self):

        # Adding component, their type (abstract, non-abstract), and their concrete implementations
        for st in self.repo_meta.subtypes.keys():
            component_name = self._prettify_name(st.cls_tpe)
            component_value = self.repo_meta.subtypes[st]

            if not isinstance(st, RepoMeta.ClassIndex):

                if component_value:
                    # If the value isn't an empty set ==> the value is an abstract component and the key is one of its concrete implementations
                    abstract_component_name = self._prettify_name(
                        list(component_value)[0].cls_tpe)
                    self.repo_dict[abstract_component_name]["abstract"] = True

                    if "concreteImplementations" in self.repo_dict[abstract_component_name]:
                        self.repo_dict[abstract_component_name]["concreteImplementations"] = \
                            self.repo_dict[abstract_component_name]["concreteImplementations"] + [component_name]
                    else:
                        self.repo_dict[abstract_component_name]["concreteImplementations"] = [component_name]

                    self.concrete_to_abstract_mapper[component_name] = abstract_component_name

                else:
                    # If the value IS an empty set ==> the key represents a (potentially!) none-abstract component
                    # If they this key appears later as a value of any given key ==> they key turns out to be in deed an abstract component
                    # and will be updated to abstract:True
                    self.repo_dict[component_name] = {"abstract": False}

        # Adding dependencies of components
        for k in self.repository.keys():
                for item in self.repository[k].organized:
                    # Path is a tuple of 2 elements; 1st element is a list of the dependencies for the component in the 2nd element
                    path = item.path
                    deps = path[0]
                    component_name = self._prettify_name(path[-1].name.cls_tpe)
                    if component_name in self.concrete_to_abstract_mapper.keys():
                        component_name = self.concrete_to_abstract_mapper[component_name]

                    for d in deps:
                        dependency = self._prettify_name(d.name.cls_tpe)

                        if dependency != component_name:
                            if "inputQueue" in self.repo_dict[component_name]:
                                if dependency not in self.repo_dict[component_name]["inputQueue"]:
                                    self.repo_dict[component_name]["inputQueue"] = self.repo_dict[component_name][
                                                                                       "inputQueue"] + [
                                                                                       dependency]
                            else:
                                self.repo_dict[component_name]["inputQueue"] = [dependency]

                if isinstance(k,  RepoMeta.ClassIndex):
                    component_name = self._prettify_name(k.cls_tpe)
                    if "configIndexes" in self.repo_dict[component_name]:
                        self.repo_dict[component_name]["configIndexes"] = self.repo_dict[component_name]["configIndexes"] + [k.at_index]
                    else:
                        self.repo_dict[component_name]["configIndexes"] = [k.at_index]


    def dump_static_repo(self, path=""):
        full_path = join(path, "static_repo.json")
        print(full_path)
        with open(full_path, "w+") as r:
            json.dump(self.repo_dict, r, indent=4)
