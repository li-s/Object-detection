# Object-localization
Detecting the positions of objects in images
For introduction, explanation, and readings on object detection, go to Image-recognition repository.

## Introduction

## Getting started

### Prerequisites
+ [Python3](https://www.python.org/download/releases/3.0/)
+ [tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup)
+ [numpy](https://www.scipy.org/scipylib/download.html)
+ [scikit-image](http://scikit-image.org/)
+ [matplotlib](http://matplotlib.org/users/installing.html)

### Usage
1. Run `get_predictions.py` to get the predictions of label, confidence, and boudning box saved in `/cache/predictions.pkl`.
2. Run `make_detfile.py` to create detection files of the images under `../data/predictions/`.
3. Run `py_faster_rcnn_evaluate.py` to get the average precision of each class and mean average precision.

or

1. Run `predict.py [image_path]` to find the prediction, confidence and bounding box of one image.

## API reference
+ [ssd keras](https://github.com/rykov8/ssd_keras)

## Miscellaneous

### Readings
+ [Single shot multibox detector research paper](https://arxiv.org/pdf/1512.02325.pdf)
