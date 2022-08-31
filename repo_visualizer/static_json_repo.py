import json
from os.path import join


class StaticJSONRepo:
    """
    Creates a json representation of all tasks in repository, their types,
    their concrete implementations (if the task is abstract), and their dependencies

    usage:
      static_repo = StaticJSONRepo(RepoMeta)
      static_repo.construct_repo_dict()
      static_repo.dump_repo(where to dump your json)
    """

    def __init__(self, repo_meta):
        self.repo_meta = repo_meta
        self.repository = repo_meta.repository
        self.concrete_to_abstract_mapper = {}  # used for mapping the concrete implementations to their abstract components
        self.repo_dict = {}

    @staticmethod
    def _prettify_name(name):
        return name.split(".")[-1]

    def construct_repo_dict(self):

        # Adding component, their type (abstract, non-abstract), and their concrete implementations
        for st in self.repo_meta.subtypes.keys():
            component_name = self._prettify_name(st.cls_tpe)
            component_value = self.repo_meta.subtypes[st]

            if component_value:
                # If the value isn't an empy set ==> the value is an abstract component and the key is one of its concrete implementations
                abstract_component_name = self._prettify_name(
                    next(iter(component_value)).cls_tpe)  # using next(iter()) to access the only element in the set
                self.repo_dict[abstract_component_name]["abstract"] = True

                if "concreteImplementations" in self.repo_dict[abstract_component_name]:
                    self.repo_dict[abstract_component_name]["concreteImplementations"] = \
                        self.repo_dict[abstract_component_name]["concreteImplementations"] + [component_name]
                else:
                    self.repo_dict[abstract_component_name]["concreteImplementations"] = [component_name]

                self.concrete_to_abstract_mapper[component_name] = abstract_component_name

            else:
                # If the value IS an empy set ==> the key represents a (potentially!) none-abstract component
                # If they this key appears later as a value of any given key ==> they key turns out to be in deed an abstract component
                # and will be updated to abstract:True
                self.repo_dict[component_name] = {"abstract": False}

        # Adding dependencies of components
        for k in self.repository.keys():
            path = self.repository[
                k].path  # path is a tuple of 2 elements; 1st element is a list of the dependencies for the component in the 2nd element
            deps = path[0]
            component_name = self._prettify_name(path[-1].name.cls_tpe)
            if component_name in self.concrete_to_abstract_mapper.keys():
                component_name = self.concrete_to_abstract_mapper[component_name]

            for d in deps:
                dependency = self._prettify_name(d.name.cls_tpe)
                if "inputQueue" in self.repo_dict[component_name]:
                    if dependency not in self.repo_dict[component_name]["inputQueue"]:
                        self.repo_dict[component_name]["inputQueue"] = self.repo_dict[component_name]["inputQueue"] + [
                            dependency]
                else:
                    self.repo_dict[component_name]["inputQueue"] = [dependency]

    def dump_repo(self, path=""):
        full_path = join(path, "repo.json")
        print(full_path)
        with open(full_path, "w+") as r:
            json.dump(self.repo_dict, r, indent=4)
