from os.path import join as pjoin
from cls_tasks import *
from tour_planning_tasks import AbstractMptopConfig

class AbstractNSConfig(AbstractMptopConfig):
    abstract: bool = True

class AbstractSABCConfig(AbstractMptopConfig):
    abstract: bool = True
    
class AbstractWABCConfig(AbstractMptopConfig):
    abstract: bool = True

class NSConfig1(AbstractMptopConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(self.global_config_path, "benchmark_ns/NS_config_1.yaml")


class NSConfig2(AbstractMptopConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(self.global_config_path, "benchmark_ns/NS_config_2.yaml")

class NSConfig3(AbstractMptopConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(self.global_config_path, "benchmark_ns/NS_config_3.yaml")

class SABCConfig1(AbstractMptopConfig):
    abstract: bool = False


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(self.global_config_path, "benchmark_sabc/sABC_config_1.yaml")

class SABCConfig2(AbstractMptopConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(self.global_config_path, "benchmark_sabc/sABC_config_2.yaml")

class SABCConfig3(AbstractMptopConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(self.global_config_path, "benchmark_sabc/sABC_config_3.yaml")

class WABCConfig1(AbstractMptopConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(self.global_config_path, "benchmark_wabc/wABC_config_1.yaml")

class WABCConfig2(AbstractMptopConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(self.global_config_path, "benchmark_wabc/wABC_config_2.yaml")

class WABCConfig3(AbstractMptopConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(self.global_config_path, "benchmark_wabc/wABC_config_3.yaml")