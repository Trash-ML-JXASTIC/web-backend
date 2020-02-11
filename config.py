import json

def read_config():
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
