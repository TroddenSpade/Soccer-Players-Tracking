from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('models/spc3.h5')

def classify(X):
    res = model.predict(X)
    return res.argmax(axis=1)

def preprocess(img):
    img = img.astype(np.float32) / 255
    resized_image = cv2.resize(img, (10, 28))
    # means = resized_image.mean(axis=(0,1), dtype='float64')
    # std = resized_image.std(axis=(0,1), dtype='float64')
    # normalized_img = (resized_image - means) / std
    return resized_image