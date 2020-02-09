from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import cgi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.nn import leaky_relu
import matplotlib.pyplot as plt
import numpy as np
import os

data = {"result": "this is a test"}
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
                img = image.load_img("upload/" + filename, target_size=(256, 256))
                os.remove("upload/" + filename)
                image_tensor = image.img_to_array(img)
                image_tensor = np.expand_dims(image_tensor, axis=0)
                image_tensor /= 255.
                pred = model.predict(image_tensor)
                pred_class = model.predict_classes(image_tensor)
                print("Raw prediction data:", pred)
                print("Raw prediction class data:", pred_class)
                response = {
                    "status": "success",
                    "result": pred_class.tolist()
                }
            elif self.path == "/train":
                with open("train/" + labels_index[form["type"].type] + filename, "wb") as f:
                    f.write(fileValue)
                response = {
                    "status": "success"
                }
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
    model = load_model("trash.h5", custom_objects={"leaky_relu": leaky_relu})
    server = HTTPServer(host, Request)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
