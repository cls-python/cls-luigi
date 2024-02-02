# -*- coding: utf-8 -*-
#
# Apache Software License 2.0
#
# Copyright (c) 2022-2023, Jan Bessai, Anne Meyer, Hadi Kutabi, Daniel Scholtyssek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import subprocess
from threading import Thread, Event
import requests
from http.server import SimpleHTTPRequestHandler, HTTPServer
import signal
import time
import sys

from cls_luigi.repo_visualizer.json_io import load_json, dump_json
VIS = os.path.dirname(os.path.abspath(__file__))

PORT = 8000
CONFIG = "config.json"
LUIGI_DAEMON_THREAD = None
UPDATE_THREAD = None

def main():
    global LUIGI_DAEMON_THREAD
    global UPDATE_THREAD

    LUIGI_DAEMON_THREAD = Thread(target=start_luigi_daemon, args=(stop_event,))
    LUIGI_DAEMON_THREAD.start()
    print("Started luigid\n")
    UPDATE_THREAD = Thread(target=update_tasks_status, args=(stop_event,))
    UPDATE_THREAD.start()
    print("Task-Status Updater is ready\n")

    try:
        os.chdir(os.path.dirname(__file__))
        httpd = HTTPServer(('', PORT), SimpleHTTPRequestHandler)
        print("\nStarted visualization server\n\nNavigate to: ",
              link('http://localhost:{}/\n\n\n'.format(PORT)))

        # Define the signal handler function
        def signal_handler(signum, frame):
            print(f"Received signal {signum}. Shutting down gracefully...")

            # Perform cleanup
            cleanup()

            # Shut down the server
            httpd.shutdown()
            httpd.server_close()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"Server running on port {PORT}. Press Ctrl+C to gracefully shut down.")

        httpd.serve_forever()


    except:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()


def link(uri, label=None):
    if label is None:
        label = uri
    parameters = ''
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'
    return escape_mask.format(parameters, uri, label)

stop_event = Event()

def start_luigi_daemon(stop_event):
    luigid_process =  subprocess.Popen(["luigid"])
    while not stop_event.is_set():
        time.sleep(1)

    luigid_process.terminate()
    luigid_process.wait()
    print("Thread is exiting!")



def update_tasks_status(stop_event):
    dynamic_repo_name = os.path.join(VIS, 'dynamic_pipeline.json')

    keep_updating = True
    while keep_updating and not stop_event.is_set():
        try:
            if os.path.exists(dynamic_repo_name):

                loaded = load_json(dynamic_repo_name)
                luigi_task_updates = requests.get("http://localhost:8082/api/task_list").json()["response"]
                tasks_status = set()

                for k in loaded.keys():
                    for j in loaded[k].keys():
                        if loaded[k][j]['luigiName'] in luigi_task_updates:
                            loaded[k][j]["status"] = luigi_task_updates[loaded[k][j]["luigiName"]]["status"]
                            loaded[k][j]["timeRunning"] = luigi_task_updates[loaded[k][j]["luigiName"]]["time_running"]
                            loaded[k][j]["startTime"] = luigi_task_updates[loaded[k][j]["luigiName"]]["start_time"]
                            loaded[k][j]["lastUpdated"] = luigi_task_updates[loaded[k][j]["luigiName"]]["last_updated"]
                            tasks_status.add(luigi_task_updates[loaded[k][j]["luigiName"]]["status"])



                dump_json(dynamic_repo_name, loaded)
                time.sleep(1) # wait a sec till system stabilizes

                # if (len(tasks_status) == 1) and ("DONE" in tasks_status):
                #     keep_updating = False
                #     print('All tasks completed \nStoping updating task status')
            else:
                raise FileNotFoundError

        except:
            # No Data Available yet
            time.sleep(3)
            pass
        finally:
            time.sleep(3)
            pass
    print("Update Thread is exiting!")

def cleanup():
    global LUIGI_DAEMON_THREAD
    global UPDATE_THREAD
    print("Shutdown CLEAN")
    stop_event.set()
    LUIGI_DAEMON_THREAD.join()
    UPDATE_THREAD.join()
    print("Main thread is exiting!")
    sys.exit()




if __name__ == '__main__':
    main()
