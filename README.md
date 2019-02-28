# FlipkartGRID_Submission
My final submission for the Flipkart GRID Challenge 2019

## APPROACH

1. We used a combination of 5 retinanet models to generate the predictions. All the models used Resnet backbones from among Resnet-18, Resnet-34 and Resnet-50.
2. There are 2 Resnet-34 and 2 Resnet-18 models each taking images of size 416x312 and 224x168(original images are of size 640x480), and a Resnet-50 model taking images of size 416x312.
3. Each model outputs the bounding box coordinates and its confidence in the prediction, for each test image. A script combines the predictions by taking only the box predicted by the model which had the highest confidence.
4. By training the model on variable input sizes and using random flips, we aimed to make our model invariant to rotation and object size.

## RUNNING THE CODE

To run the code execute the bash script CompleteScript.sh. It will train all the models and give the final predictions in root project directory.
