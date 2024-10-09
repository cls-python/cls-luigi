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

        self.server_started = False

    def start_luigi_server(self):
        try:
            self.luigi_process = subprocess.Popen(["luigid", "--background", "--logdir", self.logdir])
            self.server_started = True
            self.luigi_process.wait()
            sleep(1)
            self.logger.warning("Started Luigi daemon")


        except Exception as e:
            self.logger.warning("Could not kill Luigi daemon: {e}")
            raise e

    def shutdown_luigi_server(self):
        if self.server_started:

            try:
                self.luigi_process = subprocess.Popen(["pkill", "-f", "luigid"])
                self.luigi_process.wait()
                sleep(1)
                self.logger.warning("Killed Luigi daemon")

            except Exception as e:
                raise e
        else:
            self.logger.warning("No Luigi daemon to kill")

    def __enter__(self):
        self.start_luigi_server()
        return self

    def __exit__(self, *args):
        self.shutdown_luigi_server()
