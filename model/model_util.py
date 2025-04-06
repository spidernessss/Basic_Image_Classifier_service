from keras.applications import MobileNet
from keras.applications.regnet import preprocess_input
from keras.preprocessing import image as process_image
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras import Model
import keras
import os
import tensorflow as tf
import numpy as np
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class DeepModel:
    def __init__(self):
        self._model = self._define_image_classification_model()
        print('Loading MobileNet.')
        print()

    @staticmethod
    def _define_image_classification_model(output_layer=-1):
        print("Defining in DeepModel")
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.40 # dynamically grow the memory used on the GPU
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        print("Base Model is ", base_model)
        output = base_model.layers[output_layer].output
        output = GlobalAveragePooling2D()(output)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer= 'adam' , loss= keras.losses.binary_crossentropy, metrics=['accuracy'])
        # model.save("image.h5")
        return model

    @staticmethod
    def preprocess_image(path):
        img = process_image.load_img(path, target_size=(224, 224))
        x = process_image.img_to_array(img)
        x = preprocess_input(x)
        return x

    def extract_feature(self, img):
        features = self._model.predict(img, batch_size=1)
        return features

    def distance(self, input1, input2):
        # Cosine
        return np.dot(input1, input2.T) / \
            np.dot(np.linalg.norm(input1, axis=1, keepdims=True),
                   np.linalg.norm(input2.T, axis=0, keepdims=True))


