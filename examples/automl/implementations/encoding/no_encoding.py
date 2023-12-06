from ..template_2 import Encoding


class NoEncoding(Encoding):
    abstract = False

    def output(self):
        return self.input()
