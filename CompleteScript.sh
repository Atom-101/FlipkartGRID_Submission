#!/bin/bash

python3.6 setup.py build_ext --inplace

#sleep 3h
#Train
python3.6 -W ignore keras-retinanet-master/keras_retinanet/bin/train.py --backbone resnet34 --batch-size 16 --epochs 200 --steps 500 --lr 4e-4 --image-max-side 416 --workers 8 --no-weights csv keras_retinanet/bin/train_extended.csv keras_retinanet/bin/classes.csv

python3.6 -W ignore keras-retinanet-master/keras_retinanet/bin/train.py --backbone resnet50 --batch-size 8 --epochs 200 --steps 500 --lr 4e-4 --image-max-side 416 --workers 8 --no-weights csv keras_retinanet/bin/train_extended.csv keras_retinanet/bin/classes.csv

python3.6 -W ignore keras-retinanet-master/keras_retinanet/bin/train.py --backbone resnet18 --batch-size 128 --epochs 200 --steps 500 --lr 4e-4 --image-max-side 224 --workers 8 --no-weights csv keras_retinanet/bin/train_extended.csv keras_retinanet/bin/classes.csv

python3.6 -W ignore keras-retinanet-master/keras_retinanet/bin/train.py --backbone resnet34 --batch-size 64 --epochs 200 --steps 500 --lr 4e-4 --image-max-side 224 --workers 8 --no-weights csv keras_retinanet/bin/train_extended.csv keras_retinanet/bin/classes.csv

python3.6 -W ignore keras-retinanet-master/keras_retinanet/bin/train.py --backbone resnet18 --batch-size 32 --epochs 200 --steps 500 --lr 4e-4 --image-max-side 416 --workers 8 --no-weights csv keras_retinanet/bin/train_extended.csv keras_retinanet/bin/classes.csv


#Convert
python3.6 keras_retinanet/bin/convert_model.py resnet34-416.h5 Complete_Resnet34-416.h5
python3.6 keras_retinanet/bin/convert_model.py resnet50-416.h5 Complete_Resnet50-416.h5
python3.6 keras_retinanet/bin/convert_model.py resnet18-416.h5 Complete_Resnet18-416.h5
python3.6 keras_retinanet/bin/convert_model.py resnet34-224.h5 Complete_Resnet34-224.h5
python3.6 keras_retinanet/bin/convert_model.py resnet18-224.h5 Complete_Resnet18-224.h5


#Predict
python3.6 Predict.py 416 Complete_Resnet34-416.h5 resnet34 PredWithScores/Complete_Resnet34-416.csv
python3.6 Predict.py 224 Complete_Resnet34-224.h5 resnet34 PredWithScores/Complete_Resnet34-224.csv
python3.6 Predict.py 416 Complete_Resnet18-416.h5 resnet18 PredWithScores/Complete_Resnet18-416.csv
python3.6 Predict.py 224 Complete_Resnet18-224.h5 resnet18 PredWithScores/Complete_Resnet18-224.csv
python3.6 Predict.py 416 Complete_Resnet50-416.h5 resnet50 PredWithScores/Complete_Resnet50-416.csv


#Combine
python3.6 PredWithScores/Combine.py
