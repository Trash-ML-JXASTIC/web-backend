from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dense, Dropout, Flatten, MaxPooling2D, ZeroPadding2D, Convolution2D, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config as conf
import os
import tensorflow as tf
import time

def do_train():
	if conf.read_config()["training"] == 1:
		return

	conf.write_config("training", 1)

	tf.compat.v1.enable_eager_execution()

	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)

	train_data_root = Path("./train")

	all_train_image_paths = list(train_data_root.glob("*/*"))
	all_train_image_paths = [str(path) for path in all_train_image_paths]

	train_image_count = len(all_train_image_paths)
	print("[TRAIN] Train image count:", train_image_count)

	label_names = sorted(
		item.name for item in train_data_root.glob("*/") if item.is_dir())
	label_to_index = dict((name, index) for index, name in enumerate(label_names))

	print("[TRAIN] Label to index:", label_to_index)
	print("[TRAIN] All train image paths:", all_train_image_paths)

	BATCH_SIZE = 16
	if tf.keras.backend.image_data_format() == 'channels_first':
		input_shape = (3, 256, 256)
	else:
		input_shape = (256, 256, 3)

	output_shape = (256, 256)

	ds_input = Input(shape=input_shape, name="input")
	conv1 = Conv2D(16, (3, 3), padding="same", strides=2, activation=tf.nn.leaky_relu, name="conv1")(ds_input)
	conv2 = Conv2D(32, (3, 3), padding="same", strides=2, activation=tf.nn.leaky_relu, name="conv2")(conv1)
	conv3 = Conv2D(64, (3, 3), padding="same", strides=2, activation=tf.nn.leaky_relu, name="conv3")(conv2)
	conv4 = Conv2D(3, (3, 3), padding="same", name="conv4")(conv3)
	pool1 = GlobalAveragePooling2D(name="pool1")(conv4)
	act1 = Activation(name="act1", activation="softmax")(pool1)

	model = Model(inputs=ds_input, outputs=act1, name="model" + str(int(time.time())))

	model.compile(loss="categorical_crossentropy",
				optimizer="rmsprop", metrics=["accuracy"])

	train_datagen = ImageDataGenerator(
		rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

	train_generator = train_datagen.flow_from_directory(train_data_root, target_size=(
		output_shape), batch_size=BATCH_SIZE, class_mode='categorical')

	filepath = "model-improvement-{epoch:02d}.h5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
								save_best_only=False, save_weights_only=False, mode='auto', period=1)
	callbacks_list = [checkpoint]

	model.fit_generator(train_generator, steps_per_epoch=train_image_count // BATCH_SIZE, epochs=30, callbacks=callbacks_list)

	model.save("trash_new.h5")

	model_old = load_model("trash.h5", custom_objects={"leaky_relu": tf.nn.leaky_relu})

	print("[TRAIN] model_old:")
	model_old.summary()
	print("[TRAIN] model:")
	model.summary()

	outputs = [model_old(ds_input), model(ds_input)]
	y = tf.keras.layers.Average()(outputs)
	model_new = Model(ds_input, y, name="ensemble")
	print("[TRAIN] model_new:")
	model_new.summary()
	model_new.save("trash.h5")

	conf.write_config("training", 0)

	print("[TRAIN] Completed.")

do_train()