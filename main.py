from http.server import HTTPServer, BaseHTTPRequestHandler
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.nn import leaky_relu
import cgi
import config as conf
import json
import numpy as np
import os
import pathlib
import tensorflow as tf
import threading
import time
import train
import weight
import crop

host = ("localhost", 8080)

class Request(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_POST(self):
        if (self.path == "/get") or (self.path == "/train"):
            self.send_response(200)
        else:
            self.send_response(404)
            self.end_headers()
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD":"POST", "CONTENT_TYPE":self.headers["Content-Type"]}
        )
        field_item = form["file"]
        if (field_item == None):
            self.send_response(403)
            self.end_headers()
            return
        filename = field_item.filename
        fileValue = field_item.value
        fileSize = len(fileValue)
        print("[UPLOAD]", "file: ", filename, ", size:", len(fileValue), "bytes")
        if (len(fileValue) > 20000000) or not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".gif")):
            self.send_response(403)
            self.end_headers()
            return
        else:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-type", "application/json")
            self.end_headers()
            if self.path == "/get":
                with open("upload/" + filename, "wb") as f:
                    f.write(fileValue)
                crop.ssd_process(img_path="upload/" + filename, crop_path="crop", show=False)
                os.remove("upload/" + filename)
                files = glob.glob(crop_path + "/*")
                pred_result = []
                pred_probabilities = []
                for index, f in enumerate(files):
                    img = image.load_img(f, target_size=(256, 256))
                    image_tensor = image.img_to_array(img)
                    image_tensor = np.expand_dims(image_tensor, axis=0)
                    image_tensor /= 255.
                    model = load_model("model/trash.h5", custom_objects={"leaky_relu": leaky_relu})
                    pred = model.predict(image_tensor)
                    pred_class = np.argmax(pred, axis=1)
                    pred_result.append(pred)
                    pred_probabilities.append(pred)
                    print("[PREDICT] Crop name:", f)
                    print("[PREDICT] Raw prediction data:", pred)
                    print("[PREDICT] Raw prediction class data: ", pred_class)
                response = {
                    "status": "success",
                    "result": pred_result,
                    "probabilities": pred_probabilities
                }
            elif self.path == "/train":
                with open("train/" + labels_index[int(form["type"].value)] + "/" + filename, "wb") as f:
                    f.write(fileValue)
                response = {
                    "status": "success"
                }
                if int(time.time()) - conf.read_config()["last"] > 86400:
                    train_path = list(pathlib.Path("./train").glob("*/*"))
                    train_count = len(train_path)
                    if train_count > int(conf.read_config()["in"]):
                        conf.write_config("last", int(time.time()))
                        conf.write_config("in", weight.get_new_n(train_count))
                        train.do_train()
                        for item in train_path:
                            os.remove(item)
            responseBody = json.dumps(response)
            self.wfile.write(responseBody.encode("utf-8"))
        print()

if __name__ == "__main__":
    labels_index = [
        "cardboard",
        "plastic",
        "trash"
    ]
    tf.compat.v1.enable_eager_execution()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    server = HTTPServer(host, Request)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
