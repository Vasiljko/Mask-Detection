# Mask-Detection

Deep Learning project for detecting if a person has a mask on its face. I have used **Transfer Learning** with **VGG19** on [this dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset).
After building the model, I've used Haar Cascading for detecting faces on new images. When the face is recognized, that part of image is used for predicting if a person has a mask or not.

In order to use live face and mask detection, run *mask-detection.ipynb*  to save the model as *model.h5* and then run *live.py*

Sample test with a mask

![withmask](https://user-images.githubusercontent.com/16977953/112488821-2e78da80-8d7e-11eb-87d4-470e8bf5a1bd.PNG)

Sample test without a mask

![nomask](https://user-images.githubusercontent.com/16977953/112488989-52d4b700-8d7e-11eb-9ec1-4b4fec4a982c.PNG)
