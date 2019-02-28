import tensorflow as tf
import numpy as np
# from resnet18 import KerasResnet18,Resnet18
import pandas as pd
import keras
# from tensorflow.keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import optimizers
from keras_retinanet.models import models
# from ..models.__init__ import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from tqdm import *
import cv2

# inputs = tf.keras.Input((416,416,3))
# model = Resnet18().build_model(inputs)

df = pd.read_csv('test.csv')
# print(len(valid_df))
# valid_df['image_name'] = '/home/atom/common_data/Projects/FlipkartGRID/Fastai/images/' + valid_df['image_name']
# print(valid_df.head())
# datagen = ImageDataGenerator()
# valid_generator = datagen.flow_from_dataframe(valid_df,directory='../../../Fastai/images_test/',
#                     x_col = 'image_name',
#                     # y_col = ['x1','x2','y1','y2'],
#                     y_col = None,
#                     target_size=(416,312),batch_size=1,class_mode=None,shuffle=False)
# print(next(valid_generator)[1])

# def eval_loss(y_true,y_pred):
#     # print(y_true.shape)
#     mask = tf.ones_like(y_true)
#     mask[:,:2] *= 640
#     mask[:,2:] *= 480
#     true = tf.multiply(y_true,(np.array([640,640,480,480])*mask)
#     pred = tf.multiply(y_pred,tf.convert_to_tensor(mask))
#     return tf.reduce_mean(tf.abs(true-pred))

model = models.load_model('Level3/Model.h5',backbone_name='resnet50')
print(model.summary())
# model.compile(optimizer='adam',loss="binary_crossentropy")
# with open('model_architecture.json', 'w') as f:
#     f.write(model.to_json())
# print(model.summary())

predictions=[]
for img in tqdm(df.itertuples(index=False)):
    path = '/home/atom/common_data/Projects/FlipkartGRID/Fastai/images/'+img[0]
    image = cv2.resize(cv2.imread(path),(416,312),interpolation=cv2.INTER_AREA)
    image = preprocess_image(image)
    boxes,_,_ = model.predict_on_batch(np.expand_dims(image, axis=0))
    print(boxes)


