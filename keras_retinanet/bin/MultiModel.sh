#!/bin/bash

#python3.6 -W ignore train.py --backbone resnet34 --batch-size 16 --epochs 150 --steps 500 --lr 4e-5 --image-max-side 416 --workers 8 --snapshot snapshots/resnet34_csv_055_0.08.h5 csv train.csv classes.csv
#python3.6 -W ignore train.py --backbone resnet34 --batch-size 64 --epochs 200 --steps 100 --lr 2e-5 --image-max-side 224 --workers 8 --snapshot snapshots/resnet34_csv_033_0.21.h5 csv train.csv classes.csv
#python3.6 -W ignore train.py --backbone resnet18 --batch-size 128 --epochs 200 --steps 50 --lr 4e-4 --image-max-side 224 --workers 8 --weights Level3/Resnet18-416.h5 csv train.csv classes.csv
#python3.6 -W ignore train.py --backbone resnet50 --batch-size 8 --epochs 200 --steps 1000 --lr 4e-4 --image-max-side 416 --workers 8 --no-weights csv train.csv classes.csv
#python3.6 -W ignore train.py --backbone resnet50 --batch-size 64 --epochs 200 --steps 100 --lr 2e-5 --image-max-side 224 --workers 8 --snapshot snapshots/resnet34_csv_033_0.21.h5 csv train.csv classes.csv

sleep 2h
python3.6 -W ignore train.py --backbone resnet34 --batch-size 8 --epochs 200 --steps 500 --lr 5e-4 --image-max-side 640 --workers 8 --no-weights csv train.csv classes.csv

python3.6 train.py --backbone resnet34 --batch-size 4 --epochs 200 --steps 500 --lr 5e-4 --image-max-side 640 --snapshot snapshots/resnet34_csv_009_0.48.h5 csv train_extended.csv classes.csv


