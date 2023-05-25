import json
import os


def load_json(name):
    full_path = get_full_path(name)
    try:
        with open(full_path, 'r') as j:
            return json.load(j)
    except FileNotFoundError:
        print('File: "{}" not found'.format(full_path))
        raise FileNotFoundError


def dump_json(name, obj):
    full_path = get_full_path(name)
    with open(full_path, 'w+') as j:
        json.dump(obj, j, indent=6)


def get_full_path(name):
    return os.path.join(os.path.dirname(__file__), name)


if __name__ == "__main__":
    load_json('config.json')
