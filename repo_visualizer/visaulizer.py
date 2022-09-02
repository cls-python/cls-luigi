import os
from threading import Thread
from time import sleep
import requests
import json
from http.server import SimpleHTTPRequestHandler, HTTPServer

PORT = 8000


def main():
    luigi_daemon = Thread(target=start_luigi_daemon)
    luigi_daemon.start()
    print("Started luigid\n")
    status_update_daemon = Thread(target=update_tasks_status)
    status_update_daemon.start()
    print("Task-Status Updater is ready\n")

    try:

        httpd = HTTPServer(('', PORT), SimpleHTTPRequestHandler)
        print("\nStarted visualization server\n\n'Navigate to: ", link("http://localhost:{}/\n\n\n".format(PORT)))
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
    FILE = 'dynamic_repo.json'
    path = os.path.join(os.getcwd(), FILE)
    while True:
        try:
            with open(path, 'r') as FILE:
                loaded = json.load(FILE)

            luigi_task_updates = requests.get("http://localhost:8082/api/task_list").json()["response"]

            for k in loaded.keys():

                if loaded[k]['luigiName'] in luigi_task_updates:
                    loaded[k]["status"] = luigi_task_updates[loaded[k]["luigiName"]]["status"]

            with open(path, 'w+') as updated:
                json.dump(loaded, updated, indent=6)
        except:
            sleep(10)
            pass
        finally:
            sleep(2)
            pass


if __name__ == '__main__':
    main()

