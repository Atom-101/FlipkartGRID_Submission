-------------------------
APPROACH
-------------------------

1. We used a combination of 5 retinanet models to generate the predictions. All the models used Resnet backbones from among Resnet-18, Resnet-34 and Resnet-50.
2. There are 2 Resnet-34 and 2 Resnet-18 models each taking images of size 416x312 and 224x168(original images are of size 640x480), and a Resnet-50 model taking images of size 416x312.
3. Since the Retinanet algorithm also classifies detected objects, all objects were considered to belong to a single class. 
4. Since pretrained weights were not allowed, we trained the model on the combined training set of both Level-2 and Level-3. The dataset was further augmented using random horizontal and vertical flips.
5. Each model outputs the bounding box coordinates and its confidence in the prediction, for each test image. A script combines the predictions by taking only the box predicted by the model which had the highest confidence.
6. By training the model on variable input sizes and using random flips, we aimed to make our model invariant to rotation and object size.

---------------------------
RUNNING THE CODE
---------------------------

1. To run the code execute the bash script CompleteScript.sh. It will train all the models and give the final predictions in root project directory.


---------------------------
LIBRARIES USED
---------------------------

1. The models were built using Keras and Tensorflow.
2. For manipulating the CSV files Pandas and Numpy libraries have been used extensively.
