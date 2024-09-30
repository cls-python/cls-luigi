import logging
import subprocess
from time import sleep
from typing import Optional


class LinuxLuigiDaemonHandler:

    def __init__(
        self,
        logdir: str = "/tmp/",
        logger: Optional[logging.Logger] = None
    ) -> None:

        self.logdir = logdir
        self.luigi_process = None
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

    def start_luigi_server(self):
        try:
            self.luigi_process = subprocess.Popen(["luigid", "--background", "--logdir", self.logdir])
            self.luigi_process.wait()
            sleep(0.5)  # just waiting till system stabilizes
            self.logger.warning("Started Luigi daemon")

        except Exception as e:
            self.logger.warning("Could not kill Luigi daemon: {e}")
            raise e

    def shutdown_luigi_server(self):
        try:
            self.luigi_process = subprocess.Popen(["pkill", "-f", "luigid"])
            self.luigi_process.wait()
            sleep(0.5)  # just waiting till system stabilizes
            self.logger.warning("Killed Luigi daemon")

        except Exception as e:
            raise e
