from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('models/spc1.h5')

def classify(X):
    res = model.predict(X)
    return res.argmax(axis=1)

def preprocess(img):
    img = img.astype(np.float32) / 255
    resized_image = cv2.resize(img, (10, 28))
    return resized_image