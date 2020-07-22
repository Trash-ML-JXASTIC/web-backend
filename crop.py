import glob
import os

import tensorflow as tf
from ssd.nets import np_methods, ssd_vgg_300
from PIL import Image
from ssd.preprocessing import ssd_vgg_preprocessing

slim = tf.contrib.slim
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

net_shape = (300, 300)
data_format = "NHWC"
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

reuse = True if "ssd_net" in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

ckpt_filename = './model/ssd_300_vgg.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
ssd_anchors = ssd_net.anchors(net_shape)

def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img], feed_dict={img_input: img})
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(rpredictions, rlocalisations, ssd_anchors, select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

def ssd_process(img_path, crop_path="crop", show=False):
    print("Image to Process by SSD:", img_path)

    img = Image.open(img_path, "r")
    rclasses, rscores, rbboxes = process_image(img)
    width, height = img.size

    print("SSD Anchors:", rbboxes)

    files = glob.glob(crop_path + "/*")
    for f in files:
        os.remove(f)

    for index, box in enumerate(rbboxes):
        left = box[1] * width
        top = box[0] * height
        right = box[3] * width
        bottom = box[2] * height
        crop = img.crop((left, top, right, bottom))
        if show == True:
            crop.show()
        crop.save(crop_path + "/" + str(index) + ".jpg")
        print("Save SSD Crop:", crop_path + "/" + str(index) + ".jpg")
