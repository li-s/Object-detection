import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import os
import operator
import itertools
from collections import Counter
import pickle

from ssd import SSD300
from ssd_utils import BBoxUtility
import pascal_VOC

def numDups(a, b):
    if len(a)>len(b):
        a,b = b,a

    a_count = Counter(a)
    b_count = Counter(b)

    return sum(min(b_count[ak], av) for ak,av in a_count.items())

def load_model():
    # matplotlib inline
    plt.rcParams['figure.figsize'] = (8, 8)
    plt.rcParams['image.interpolation'] = 'nearest'

    np.set_printoptions(suppress=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    set_session(tf.Session(config=config))

    voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                   'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                   'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    NUM_CLASSES = len(voc_classes) + 1

    input_shape=(300, 300, 3)
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('../data/weights_SSD300.hdf5', by_name=True)
    bbox_util = BBoxUtility(NUM_CLASSES)

    return model, bbox_util

def validates_images(model, bbox_util):
    inputs = []
    images = []
    files = []
    for filename in os.listdir('../../data/VOC2007/JPEGImages'):
        if filename.endswith('.jpg'):
            files.append(filename)

    b =0
    for filename in sorted(files):
        if b < 3:
            img_path = '../../data/VOC2007/JPEGImages/' + filename
            img = image.load_img(img_path, target_size=(300, 300))
            img = image.img_to_array(img)
            images.append(imread(img_path))
            inputs.append(img.copy())
            b += 1

    inputs = preprocess_input(np.array(inputs))
    preds = model.predict(inputs, batch_size=1, verbose=1)

    results = bbox_util.detection_out(preds)

    return results, img

def process_images(results):
    image_list = []
    for i in range(len(results)):
        a_list = []
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.4, as it gives the highest % accuracy of labels.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.4]

        # Format of a_list: confidence, labels, xmin, ymin, xmax, ymax
        a_list.append(det_conf[top_indices])
        a_list.append(det_label[top_indices].tolist())
        a_list.append(det_xmin[top_indices])
        a_list.append(det_ymin[top_indices])
        a_list.append(det_xmax[top_indices])
        a_list.append(det_ymax[top_indices])

        image_list.append(a_list)

    return image_list

def checker(image_list, img):
    for i in range(len(image_list)):
        for j in range(image_list[i][0].shape[0]):
            if j < 1:
                xmin = int(round(image_list[i][2][j] * img.shape[1]))
                ymin = int(round(image_list[i][3][j] * img.shape[0]))
                xmax = int(round(image_list[i][4][j] * img.shape[1]))
                ymax = int(round(image_list[i][5][j] * img.shape[0]))
                print(xmin, ymin, xmax, ymax)
                print(image_list[i][2][j], image_list[i][3][j], image_list[i][4][j], image_list[i][5][j])
                input('1')

    with open('../data/VOC2007.pkl', 'rb') as read:
        x = pickle.load(read)
        sorted_x = sorted(x.items(), key=operator.itemgetter(0))
        number_correct = 0
        total = 0
        extras = 0
        a = 0
        for i, j in enumerate(sorted_x):
            if a < 3:
                print(j[1])
                list_of_one_hot = [[k for k, int1 in enumerate(a_list[3:]) if int1 == 1.0] for a_list in j[1]]
                list_of_one_hot = list(itertools.chain.from_iterable(list_of_one_hot))
                # This counts how many labels there are in total
                total += len(list_of_one_hot)
                # This counts how many labels are correct
                similarity = numDups(image_list[i][1], list_of_one_hot)
                number_correct +=similarity
                # This counts how many extra labels are identified
                extras += len(image_list[1][i])
                a+=1

    return number_correct, total, extras

if __name__ == '__main__':
    model, bbox_util = load_model()
    results, img = validates_images(model, bbox_util)
    image_list = process_images(results)
    number_correct, total, extras = checker(image_list, img)
    if extras > total:
        extras -= total
    else:
        extras = 0
    percentage = (number_correct - extras)/total*100
    print('Total: {}\nCorrect: {}\nExtras: {}\nPercentage: {}%'.format(total, number_correct, extras, percentage))
