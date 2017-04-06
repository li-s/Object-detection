import ml_metrics as metrics
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.backend.tensorflow_backend import set_session
from scipy.misc import imread
from keras.applications.imagenet_utils import preprocess_input
import pickle

from ssd import SSD300
from ssd_utils import BBoxUtility

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def get_ground_truth(image_file):
    with open(image_file, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # Gets ground truth boxes with independent list for each image.
    ground_truth = []
    for img in imagenames:
        filename = '../data/VOC2007/Annotations/' + img + '.xml'
        objects = parse_rec(filename)
        for i in objects:
            file_temp = []
            file_temp.append(i['name'])
            file_temp.append(i['bbox'])
            ground_truth.append(file_temp)

    ground_truth = tuple(ground_truth)

    return imagenames, ground_truth

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

    return model, bbox_util, voc_classes

def validates_images(imagenames, model, bbox_util):
    inputs = []
    images = []
    for filename in imagenames:
        img_path = '../data/VOC2007/JPEGImages/' + filename + '.jpg'
        img = image.load_img(img_path, target_size=(300, 300))
        img = image.img_to_array(img)
        images.append(imread(img_path))
        inputs.append(img.copy())

    inputs = preprocess_input(np.array(inputs))
    preds = model.predict(inputs, batch_size=1, verbose=1)

    results = bbox_util.detection_out(preds)

    return images, results

def process_images(images, results, voc_classes):
    predictions = []
    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.4, as it gives the highest % accuracy of labels.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.4]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        temp_predictions = []
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = voc_classes[label - 1]

            temp_predictions.append([label_name, score, xmin, ymin, xmax, ymax])
        predictions.append(temp_predictions)

    return predictions

if __name__ == '__main__':
    # Get images to validate on
    imagenames, ground_truth = get_ground_truth('../data/VOC2007/ImageSets/Layout/val.txt')

    # Load model
    model, bboxutil, voc_classes = load_model()

    # Perform prediction
    images, results= validates_images(imagenames, model, bboxutil)

    # Get bounding box and label of predictions
    predictions = process_images(images, results, voc_classes)
    with open('cache/predictions.pkl', 'wb') as w:
        pickle.dump(predictions, w)
