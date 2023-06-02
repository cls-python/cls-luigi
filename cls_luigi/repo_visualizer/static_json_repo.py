from .json_io import load_json, dump_json
import cls.types
CONFIG = "config.json"
import os

VIS = os.path.dirname(os.path.abspath(__file__))




class StaticJSONRepo:
    def __init__(self, repo_meta):
        self.repo_meta = repo_meta
        self.output_dict = {}
        self.possibly_abs, self.possibly_not_abs = [], []
        self.subclasses_to_bases = {}
        self.not_abs = []

        self.differentiate_tasks()
        self.map_subclasses_to_bases()
        self.add_nodes_to_output_dict()
        self.add_base_classes_to_output_dict()
        self.add_dependencies_to_output_dict()

        self.config = load_json(CONFIG)
        if os.path.exists(os.path.join(VIS, self.config['static_pipeline'])):
            os.remove(os.path.join(VIS, self.config['static_pipeline']))

    @staticmethod
    def _prettify_name(name):
        return name.split(".")[-1]

    def differentiate_tasks(self):

        for k in self.repo_meta.subtypes.keys():
            if k.tpe.__subclasses__():
                self.possibly_abs.append(k)
            else:
                self.possibly_not_abs.append(k)
                if self.repo_meta.subtypes[k]:
                    self.not_abs.append(k)

    def map_subclasses_to_bases(self):

        for t in self.possibly_abs:
            if t.tpe.abstract is not None:
                # assert t.tpe.abstract is True, f"{t} should be an abstract class"
                pretty_bas_task_name = self._prettify_name(t.cls_tpe)
                for s in t.tpe.__subclasses__():
                    self.subclasses_to_bases[self._prettify_name(s.__name__)] = pretty_bas_task_name

    def add_nodes_to_output_dict(self):
        for t in self.possibly_abs:
            if t.tpe.abstract is not None:
                if (t.tpe.__subclasses__()) and (t.tpe.abstract is True):
                    abstract = True
                else:
                    abstract = False
                pretty_task_name = self._prettify_name(t.cls_tpe)
                self.output_dict[pretty_task_name] = {"abstract": abstract}

        for t in self.not_abs:
            # assert len(t.tpe.__subclasses__()) == 0, f"Not abstract task {t} has subclasses"

            pretty_task_name = self._prettify_name(t.cls_tpe)

            if pretty_task_name not in self.subclasses_to_bases:
                self.output_dict[pretty_task_name] = {"abstract": False}

            else:
                abs_task = self.subclasses_to_bases[pretty_task_name]
                if "concreteImplementations" not in self.output_dict[abs_task]:
                    self.output_dict[abs_task]["concreteImplementations"] = []
                if pretty_task_name not in self.output_dict[abs_task]["concreteImplementations"]:
                    self.output_dict[abs_task]["concreteImplementations"].append(
                        pretty_task_name
                    )

    def add_base_classes_to_output_dict(self):

        for t in self.possibly_abs:
            pretty_name = self._prettify_name(t.cls_tpe)

            if pretty_name in self.subclasses_to_bases:
                if self.output_dict[pretty_name]["abstract"] is True:
                    base = self.subclasses_to_bases[pretty_name]
                    if "baseOf" not in self.output_dict[base]:
                        self.output_dict[base]["baseOf"] = []
                    if pretty_name not in self.output_dict[base]["baseOf"]:
                        self.output_dict[base]["baseOf"].append(pretty_name)

    def add_dependencies_to_output_dict(self):

        for k, v in self.repo_meta.repository.items():
            if not isinstance(v, cls.types.Constructor):
                predecessors = v.path
                pretty_task = self._prettify_name(k.name)

                if k.has_index is False:
                    if predecessors:
                        for p in predecessors[0]:
                            pretty_p = self._prettify_name(p.name.cls_tpe)

                            if pretty_task in self.subclasses_to_bases:
                                if "baseOf" not in self.output_dict[self.subclasses_to_bases[pretty_task]]:
                                    pretty_task = self.subclasses_to_bases[pretty_task]

                            if "inputQueue" not in self.output_dict[pretty_task]:
                                self.output_dict[pretty_task]["inputQueue"] = []

                            if pretty_p not in self.output_dict[pretty_task]["inputQueue"]:
                                if "baseOf" not in self.output_dict[pretty_task]["inputQueue"]:
                                    self.output_dict[pretty_task]["inputQueue"].append(pretty_p)
                else:
                    intersection = v
                    inter_org = list(intersection.organized)
                    for ix, i in enumerate(inter_org):
                        path = i.path
                        index = path[0][0].name.at_index
                        indexed_t_name = path[0][1].name.cls_tpe
                        pretty_indexed_t_name = self._prettify_name(indexed_t_name)

                        if self.output_dict.get(pretty_task) is None:
                            print(pretty_task)
                            pretty_task = self.subclasses_to_bases[pretty_task]

                        if "configIndexes" not in self.output_dict[pretty_task]:
                            self.output_dict[pretty_task]["configIndexes"] = {}
                        self.output_dict[pretty_task]["configIndexes"][index] = [pretty_indexed_t_name]

                        if pretty_indexed_t_name in self.subclasses_to_bases:
                            pretty_indexed_t_name = self.subclasses_to_bases[pretty_indexed_t_name]

                        if "inputQueue" not in self.output_dict[pretty_task]:
                            self.output_dict[pretty_task]["inputQueue"] = []

                        self.output_dict[pretty_task]["inputQueue"] += [pretty_indexed_t_name]
    def dump_static_repo_json(self):
        outfile_name = load_json(CONFIG)['static_pipeline']
        dump_json(outfile_name, self.output_dict)
