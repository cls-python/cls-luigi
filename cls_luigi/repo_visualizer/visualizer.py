import os
from threading import Thread
from time import sleep
import requests
from http.server import SimpleHTTPRequestHandler, HTTPServer

from .json_io import load_json, dump_json

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
    dynamic_repo_name = load_json(CONFIG)['dynamic_pipeline']

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

            if (len(tasks_status) == 1) and ("DONE" in tasks_status):
                keep_updating = False
                print('All tasks completed \nStoping updating task status')

        except:
            print("No Data Available yet\n ")
            sleep(5)
            pass
        finally:
            sleep(5)
            pass


if __name__ == '__main__':
    main()
