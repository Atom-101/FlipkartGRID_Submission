import tensorflow as tf
import numpy as np
# from resnet18 import KerasResnet18,Resnet18
import pandas as pd
import keras
# from tensorflow.keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import optimizers
from keras_retinanet.models import *
# from ..models.__init__ import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from tqdm import tqdm
import cv2
import sys

mode = sys.argv[1]
model_path = sys.argv[2]
backbone = sys.argv[3]
save_path = sys.argv[4]

df = pd.read_csv('Level3/test.csv')

model = load_model(model_path,backbone_name=backbone)
print(model.summary())

predictions=[]
boxes,scores = None,[]
for img in tqdm(df.itertuples(index=False)):
# for _ in range(1):
    path = '/home/atom/common_data/Projects/FlipkartGRID/Dataset/images/'+img[0]
    
    #######
    ##214##
    #######    
    if(mode=='224'):    
        image = cv2.resize(cv2.imread(path),(224,168),interpolation=cv2.INTER_AREA)
        # image = cv2.resize(cv2.imread('/home/atom/common_data/Projects/FlipkartGRID/Dataset/images/1470734436503DSC_0138.png'),(224,168),interpolation=cv2.INTER_AREA)
        image = preprocess_image(image,mode='tf')
        boxes,score,_ = model.predict_on_batch(np.expand_dims(image, axis=0))
        predictions.append(boxes[0,np.where(np.max(score))[0],:]*640.0/224.0)
        scores.append(np.max(score)) 
    #######

    #######
    ##416##
    #######
    elif(mode=='416'):
        image = cv2.resize(cv2.imread(path),(416,312),interpolation=cv2.INTER_AREA)
        # print(image.shape)
        image = preprocess_image(image,mode='tf')
        boxes,score,_ = model.predict_on_batch(np.expand_dims(image, axis=0))
        predictions.append(boxes[0,np.where(np.max(score))[0],:]*640.0/416.0) 
        scores.append(np.max(score))   
    #######
    
    # print(boxes)

predictions = np.array(predictions)
scores = np.array(scores)
df['x1'] = predictions[:,0,0]
df['x2'] = predictions[:,0,2]
df['y1'] = predictions[:,0,1]
df['y2'] = predictions[:,0,3]
df['scores'] = scores

df['x1'] = df['x1'].astype(int)
df['x2'] = df['x2'].astype(int)
df['y1'] = df['y1'].astype(int)
df['y2'] = df['y2'].astype(int)

df.to_csv(save_path,index=False)