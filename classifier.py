#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Bashir Kazimi"
__email__ = "kazimibashir907@gmail.com"
__website__ = "https://bashirkazimi.github.io/"

from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input, MobileNetV2
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np

img_height = 192
img_width = 192

def get_data_augmentation():
  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomZoom((0.2,0.3)),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
  ])
  return data_augmentation

def create_model():

    IMG_SHAPE = (img_height,img_width) + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    base_model.trainable = False
    prediction_layer = tf.keras.layers.Dense(17,activation='sigmoid')
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = get_data_augmentation()(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
bootstrap = Bootstrap(app)
model = create_model()
model.load_weights("static/weights/weights")
CLASS_LABELS = ['mammo - mlo',
 'mammo - mag cc',
 'mammo - mag xcc',
 'mammo - xcc',
 'longitudinal',
 'lateral',
 'frontal',
 'pa',
 'oblique',
 'transverse',
 'mammo - cc',
 'coronal',
 'axial',
 'decubitus',
 'sagittal',
 'ap',

 '3d reconstruction']

class UploadForm(FlaskForm):
    upload = FileField('Select an image:', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png', 'jpeg', 'JPEG', 'PNG', 'JPG'], 'Images only!')
    ])
    submit = SubmitField('Classify')

def map_labels(label_int):
 return {label_int:name for label_int,name  in enumerate(sorted(CLASS_LABELS))}[label_int]

def get_prediction(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (192, 192))
    img_preprocessed = preprocess_input(img_resized)
    eval_ds = tf.data.Dataset.from_tensor_slices([[img_preprocessed]])
    prediction = model.predict(eval_ds)
    y_pred = np.argmax(prediction, axis=1)
    return map_labels(y_pred[0])

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.upload.data
        filename = secure_filename(f.filename)
        file_url = os.path.join('static', filename
        )
        f.save(file_url)
        form = None
        prediction = get_prediction(file_url)
    else:
        file_url = None
        prediction = None
    return render_template("index.html", form=form, file_url=file_url, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)