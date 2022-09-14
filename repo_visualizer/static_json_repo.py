from inhabitation_task import RepoMeta
from repo_visualizer.json_io import load_json, dump_json

CONFIG = "config.json"


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
        for key in self.repo_meta.subtypes.keys():
            if not isinstance(key, RepoMeta.ClassIndex):

                component_name = self._prettify_name(key.cls_tpe)
                value = self.repo_meta.subtypes[key]

                if value:
                    # If the value isn't an empty set ==> the value is an abstract component and the key is one of its concrete implementations
                    abstract_component_name = self._prettify_name(
                        list(value)[0].cls_tpe)
                    self.repo_dict[abstract_component_name]["abstract"] = True

                    self.repo_dict[abstract_component_name]["concreteImplementations"] = \
                    self.repo_dict[abstract_component_name]["concreteImplementations"] + \
                    [component_name] if "concreteImplementations" in self.repo_dict[abstract_component_name] else [
                        component_name]

                    self.concrete_to_abstract_mapper[component_name] = abstract_component_name

                else:
                    # If the value IS an empty set ==> the key represents a (potentially!) none-abstract component
                    # If they this key appears later as a value of any given key ==> they key turns out to be in deed an abstract component
                    # and will be updated to abstract:True
                    self.repo_dict[component_name] = {"abstract": False}

        # Adding dependencies of components
        for k in self.repository.keys():
            if not isinstance(k, RepoMeta.ClassIndex):
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
                                                                                       "inputQueue"] + [dependency]
                            else:
                                self.repo_dict[component_name]["inputQueue"] = [dependency]

                if k.has_index:
                    item = list(self.repository[k].organized)
                    component_name = self._prettify_name(k.name)
                    self.repo_dict[component_name]["configIndexes"] = {}

                    for c in item:
                        path = c.path[0]
                        index = path[0].name.at_index
                        for i in path[1:]:
                            indexed_task_name = self._prettify_name(i.name.cls_tpe)
                            if index in self.repo_dict[component_name]["configIndexes"]:
                                self.repo_dict[component_name]["configIndexes"][index] = \
                                self.repo_dict[component_name]["configIndexes"][index] + \
                                [indexed_task_name]
                            else:
                                self.repo_dict[component_name]["configIndexes"][index] = [indexed_task_name]

    def dump_static_repo_json(self):
        outfile_name = load_json(CONFIG)['static_repo']
        dump_json(outfile_name, self.repo_dict)