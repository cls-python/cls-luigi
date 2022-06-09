from collections.abc import Iterable


class UniqueTaskPipelineValidator(object):

    def __init__(self, unique_abstract_classes):
        self.unique_abstract_classes = unique_abstract_classes

    def validate(self, pipeline):
        traversal = self.dfs(pipeline)
        concrete_cls_map = dict()
        for obj in traversal:
            for cls in self.unique_abstract_classes:
                if isinstance(obj, cls):
                    if cls in concrete_cls_map:
                        concrete_cls = concrete_cls_map[cls]
                        if not isinstance(obj, concrete_cls):
                            return False
                    else:
                        concrete_cls_map[cls] = obj.__class__
        return True

    def dfs(self, start):
        traversal = [start]
        dependencies = start.requires()
        if isinstance(dependencies, Iterable):
            for dependency in start.requires():
                traversal.extend(self.dfs(dependency))
        else:
            traversal.extend(self.dfs(dependencies))

        return traversal
