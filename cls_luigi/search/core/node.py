import abc


class NodeBase(abc.ABC):
    def __init__(
        self,
        **kwargs
    ) -> None:
        ...

