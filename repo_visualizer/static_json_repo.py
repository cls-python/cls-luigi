from inhabitation_task import RepoMeta
from repo_visualizer.json_io import load_json, dump_json

CONFIG = "config.json"


class StaticJSONRepo:
    """
    Creates a json representation of all components in repository, their types,
    their concrete implementations or config indexes, and their dependencies.

    usage:
      StaticJSONRepo(RepoMeta).dump_static_repo_json()
    """

    def __init__(self, repo_meta):
        self.repo_meta = repo_meta
        self.repository = repo_meta.repository
        self.concrete_to_abstract_mapper = {}  # used for mapping the concrete implementations to their abstract components
        self.repo_dict = {}

        self._construct_repo_dict()

    def _construct_repo_dict(self):

        for key in self.repo_meta.subtypes.keys():
            if not isinstance(key, RepoMeta.ClassIndex):
                self._add_components_to_repo_dict(key)

        # Adding dependencies of components
        for repo_key in self.repository.keys():
            if not isinstance(repo_key, RepoMeta.ClassIndex):
                for item in self.repository[repo_key].organized:
                    self._add_deps(item)

                if repo_key.has_index:
                    self._add_config_indexes(repo_key)

    @staticmethod
    def _prettify_name(name):
        return name.split(".")[-1]

    def _add_components_to_repo_dict(self, subtype_key):
        # Adding component, their type (abstract, non-abstract), and their concrete implementations
        component_name = self._prettify_name(subtype_key.cls_tpe)
        component_value = self.repo_meta.subtypes[subtype_key]

        if component_value:
            # If the value isn't an empty set ==> the value is an abstract component and the key is one of its concrete implementations
            abs_component_name = self._prettify_name(
                list(component_value)[0].cls_tpe)
            self.repo_dict[abs_component_name]["abstract"] = True

            self.repo_dict[abs_component_name]["concreteImplementations"] = \
                self.repo_dict[abs_component_name]["concreteImplementations"] + \
                [component_name] if "concreteImplementations" in self.repo_dict[abs_component_name] else [component_name]

            self.concrete_to_abstract_mapper[component_name] = abs_component_name

        else:
            # If the value IS an empty set ==> the key represents a (potentially!) none-abstract component
            # If they this key appears later as a value of any given key ==> they key turns out to be in deed an abstract component
            # and will be updated to abstract:True
            self.repo_dict[component_name] = {"abstract": False}

    def _add_deps(self, item):
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

    def _add_config_indexes(self, repo_key):
        item = list(self.repository[repo_key].organized)
        component_name = self._prettify_name(repo_key.name)
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
        outfile_name = load_json(CONFIG)['static_pipeline']
        dump_json(outfile_name, self.repo_dict)
