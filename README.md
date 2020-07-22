# Web Backend of Trash-ML-JXASTIC

ML-based Trash Classification Project for JXASTIC Contest

TensorFlow 1.15 is required.

Directory ```/ssd``` is adapted from ```/nets```, ```/preprocessing```, and ```/tf_extended``` of ```balancap/SSD-Tensorflow```. To use, please change all imports of ```nets```, ```preprocessing```, and ```tf_extended``` to ```ssd.nets```, ```ssd.preprocessing```, and ```ssd.tf_extended```.

Directory ```/model``` contains ```trash.h5```, ```ssd_300_vgg.ckpt.data-00000-of-00001```, and ```ssd_300_vgg.ckpt.index```. The trash classification model can be trained by ```Trash-ML-JXASTIC/ML-backend```, and the SSD model can be trained by ```balancap/SSD-Tensorflow```.
