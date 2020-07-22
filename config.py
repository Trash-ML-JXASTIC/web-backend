import json
import os

def init_config():
    temp = {
        "training": 0,
        "last": 0,
        "in": 997
    }
    w = json.dumps(temp)
    with open("data.json", "w") as f:
        f.write(w)

def read_config():
    if (not os.path.exists("model/trash.h5")):
        init_config()
    with open("data.json", "r") as f:
        for line in f:
            temp = json.loads(line)
    return temp

def write_config(key, value):
    temp = read_config()
    temp[key] = value
    w = json.dumps(temp)
    with open("data.json", "w") as f:
        f.write(w)
