import config

def calc_weight(i, t):
	return i / (i + t)

def get_new_n(u):
	o = config.read_config()["in"]
	return int(o * calc_weight(o, u) + u)
