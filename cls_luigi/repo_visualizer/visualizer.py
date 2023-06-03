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
from threading import Thread
from time import sleep
import requests
from http.server import SimpleHTTPRequestHandler, HTTPServer

from cls_luigi.repo_visualizer.json_io import load_json, dump_json
VIS = os.path.dirname(os.path.abspath(__file__))

PORT = 8000
CONFIG = "config.json"

def main():
    luigi_daemon = Thread(target=start_luigi_daemon)
    luigi_daemon.start()
    print("Started luigid\n")
    status_update_daemon = Thread(target=update_tasks_status)
    status_update_daemon.start()
    print("Task-Status Updater is ready\n")

    try:
        os.chdir(os.path.dirname(__file__))
        httpd = HTTPServer(('', PORT), SimpleHTTPRequestHandler)
        print("\nStarted visualization server\n\nNavigate to: ",
              link('http://localhost:{}/\n\n\n'.format(PORT)))
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


def start_luigi_daemon():
    os.system("luigid")


def update_tasks_status():
    dynamic_repo_name = os.path.join(VIS, 'dynamic_pipeline.json')

    keep_updating = True
    while keep_updating:
        try:
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
            sleep(1) # wait a sec till system stabilizes

            # if (len(tasks_status) == 1) and ("DONE" in tasks_status):
            #     keep_updating = False
            #     print('All tasks completed \nStoping updating task status')

        except:
            print("No Data Available yet\n ")
            sleep(3)
            pass
        finally:
            sleep(3)
            pass


if __name__ == '__main__':
    main()
