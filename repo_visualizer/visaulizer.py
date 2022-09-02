import os
from threading import Thread
from time import sleep
import requests
import json
import http.server
import socketserver


def main(PORT=8000):
    status_update_daemon = Thread(target=update_tasks_status, daemon=True)
    status_update_daemon.start()

    h = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), h) as httpd:
        print("Navigate to: ", link("http://localhost:{}/".format(PORT)))
        httpd.serve_forever()


def link(uri, label=None):
    if label is None:
        label = uri
    parameters = ''

    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'

    return escape_mask.format(parameters, uri, label)


def update_tasks_status(file='dynamic_repo.json'):
    path = os.path.join(os.getcwd(), file)
    while True:
        try:
            with open(path, 'r') as file:
                loaded = json.load(file)

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
