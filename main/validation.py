abimport cv2
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

from ssd import SSD300
from ssd_utils import BBoxUtility

def numDups(a, b):
    if len(a)>len(b):
        a,b = b,a

    a_count = Counter(a)
    b_count = Counter(b)

    return sum(min(b_count[ak], av) for ak,av in a_count.items())

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

inputs = []
images = []
files = []
for filename in os.listdir('../data/VOC2007/JPEGImages'):
    if filename.endswith('.jpg'):
        files.append(filename)

b =0
for filename in sorted(files):
    if b < 10:
        img_path = '../data/VOC2007/JPEGImages/' + filename
        img = image.load_img(img_path, target_size=(300, 300))
        img = image.img_to_array(img)
        images.append(imread(img_path))
        inputs.append(img.copy())
        b += 1

inputs = preprocess_input(np.array(inputs))
preds = model.predict(inputs, batch_size=1, verbose=1)

results = bbox_util.detection_out(preds)

label = []
for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.4]

    top_conf = det_conf[top_indices]
    label.append(det_label[top_indices].tolist())

import pickle
with open('../data/VOC2007.pkl', 'rb') as read:
    x = pickle.load(read)
    sorted_x = sorted(x.items(), key=operator.itemgetter(0))
    number_correct = 0
    total = 0
    extras = 0
    a = 0
    for i, j in enumerate(sorted_x):
        if a < 10:
            list_of_one_hot = [[k for k, int1 in enumerate(a_list[3:]) if int1 == 1.0] for a_list in j[1]]
            list_of_one_hot = list(itertools.chain.from_iterable(list_of_one_hot))
            # This counts how many labels there are in total
            total += len(list_of_one_hot)
            # This counts how many labels are correct
            similarity = numDups(label[i], list_of_one_hot)
            number_correct +=similarity
            # This counts how many extra labels are identified
            extras += len(label[i])
            a+=1

    if extras > total:
        extras -= total
    else:
        extras = 0
    percentage = (number_correct - extras)/total*100
    print('Total: {}\nCorrect: {}\nExtras: {}\nPercentage: {}%'.format(total, number_correct, extras, percentage))
