from ..template import CategoryCoalescer


class NoCategoryCoalescence(CategoryCoalescer):
    abstract = False

    def output(self):
        return self.input()
