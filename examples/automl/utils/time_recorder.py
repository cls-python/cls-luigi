import json
from time import time
from time import sleep
import os


class TimeRecorder(object):
    def __init__(self, out_file_path: str) -> None:
        self.out_file_path = out_file_path
        self.start_time = None
        self.end_time = None

    def __enter__(self) -> 'TimeRecorder':
        self.start_time = time()
        return self

    def __exit__(self, *args) -> None:
        self.checkpoint()

    def checkpoint(self, message: str = "") -> None:
        self.end_time = time()
        self._write_elapsed_time_with_message(message)

    def _write_elapsed_time_with_message(self, message):
        if not message:
            message = "total_seconds"

        output_dict = {
            message: self.end_time - self.start_time
        }

        if os.path.exists(self.out_file_path):
            with open(self.out_file_path, "r") as existing_json:
                existing_json = json.load(existing_json)
            output_dict.update(existing_json)

        with open(self.out_file_path, "w") as out_file:
            json.dump(output_dict, out_file, indent=3)


if __name__ == "__main__":
    with TimeRecorder("test.json") as tr:
        tr.checkpoint("first")
        sleep(3)
        tr.checkpoint("second")
        sleep(3)
