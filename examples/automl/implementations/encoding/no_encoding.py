from ..template import CategoricalEncoder


class NoEncoding(CategoricalEncoder):
    abstract = False

    def output(self):
        return self.input()
