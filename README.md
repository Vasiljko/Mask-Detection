# Mask-Detection

Deep Learning project for detecting if a person has a mask on its face. I have used **Transfer Learning** with **VGG19** on [this dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset).
After building the model, Haar Cascading is used for detecting faces on new images. When the face is recognized, that part of image is used for predicting if a person has a mask or not.

In order to use live face and mask detection, run *mask-detection.ipynb* file first and then *run live.py*
