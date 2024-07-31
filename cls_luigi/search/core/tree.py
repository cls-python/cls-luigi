import abc
import logging

from cls_luigi.search.core.node import NodeBase


class TreeBase(abc.ABC):
    def __init__(
        self,
        root: NodeBase,
        logger: logging.Logger = None,
        ** kwargs
    ) -> None:

            if logger:
                self.logger = logger
            else:
                self.logger = logging.getLogger(self.__class__.__name__)

            self.root = root
