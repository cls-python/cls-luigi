import json
import pickle
from typing import Dict, Any


def load_json(path):
    with open(path, 'r') as j:
        return json.load(j)


def dump_json(path: str, obj: Dict[Any, Any], indent=6):

    with open(path, 'w+') as j:
        json.dump(obj, j, indent=indent)


def dump_pickle(obj: Any, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


