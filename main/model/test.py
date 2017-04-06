import ml_metrics as metrics
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.backend.tensorflow_backend import set_session
from scipy.misc import imread
from keras.applications.imagenet_utils import preprocess_input

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
        filename = '../../data/VOC2007/Annotations/' + img + '.xml'
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
    model.load_weights('../../data/weights_SSD300.hdf5', by_name=True)
    bbox_util = BBoxUtility(NUM_CLASSES)

    return model, bbox_util, voc_classes

def validates_images(imagenames, model, bbox_util):
    inputs = []
    images = []
    for filename in imagenames:
        img_path = '../../data/VOC2007/JPEGImages/' + filename + '.jpg'
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

def bb_intersection_over_union(boxA, boxB):
    '''
    boxA/B : ground truth bounding box and validation box in any order (a bounding box)
    '''

    print('boxA: {} boxB: {}\nboxA[i]: {} boxB[i]: {}'.format(boxA, boxB, boxA[i], boxB[i]))
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = float(interArea / float(boxAArea + boxBArea - interArea))

    # return the intersection over union value
    return iou

def seperate(ground_truth):
    # Formats data
    # truth = name of object
    # gt_box = coordinates of bounding box
    truth = []
    gt_box = []
    for a in ground_truth:
        temp_truth = []
        temp_gt_box = []
        for i, j in enumerate(a):
            if i%2 == 0:
                temp_truth.append(j)
            else:
                temp_gt_box.append(j)
        truth.append(temp_truth)
        gt_box.append(temp_gt_box)
    return truth, gt_box

def ap_iou_preprocess(classname, ground_truth, predictions):
    '''
    classname = name of the object to get iou and calculate average precision (one string)
    ground_truth = ground truth (tuple, containing list of label and bounding box)
    predictions = predictions (list of lists with format [label, xxx, bounding box coordinates])
    '''
    # Process predictions to be passed on to perform iou and calculation of average precision
    predict = []
    for prediction in predictions:
        if prediction[0] == classname:
            current_class.append(prediction)

    truth = []
    for i in ground_truth:
        if i[0] == classname:
            truth.append(i)

    # for i in range(len())

# DONT USE
def conform(truth, predictions):
    # Make number of predictions same as ground truth
    for i in range(len(truth)):
        predictions[i].sort(key=lambda x: x[1], reverse = True)
        if len(predictions[i]) > len(truth[i]):
            number_to_del = len(predictions[i]) - len(truth[i])
            del predictions[i][number_to_del:]
        elif len(truth[i]) > len(predictions[i]):
            predictions[i].append(['dog',0,0,0,0,0])

    predict_label = []
    predict_box = []
    for i in range(len(predictions)):
        temp_predict_label = []
        temp_predict_box = []
        for j in range(len(predictions[i])):
            temp_predict_label.append(predictions[i][j][0])
            temp_predict_box.append(predictions[i][j][2:])
        predict_label.append(temp_predict_label)
        predict_box.append(temp_predict_box)
    return predict_label, predict_box

if __name__ == '__main__':
    # Get images to validate on
    imagenames, ground_truth = get_ground_truth('../../data/VOC2007/ImageSets/Layout/val.txt')
    # Works

    # Seperate boudning box from label of ground truth
    # truth, gt_box = seperate(ground_truth)
    # Works

    # Load model
    model, bboxutil, voc_classes = load_model()
    # Works

    # Perform prediction
    images, results= validates_images(imagenames, model, bboxutil)
    # Works

    # Get bounding box and label of predictions
    predictions = process_images(images, results, voc_classes)
    print(predictions)
    input('1')
    # Works

    ap_iou_preprocess(classname, ground_truth, predictions)

    # Perform IOU
    iou = bb_intersection_over_union(gt_box, predict_box)

    # Perform average precision calculations
    ap = metrics.mapk(truth, predict_label, 328) #may need to minus 1
    print('IoU: {}\nAP: {}'.format(iou, ap))






'''
Keeping for reference
'''

# from collections import namedtuple
#
# # define the `Detection` object
# Detection = namedtuple("Detection", ["image_path", "gt", "pred"])
#
# # define the list of example detections
# examples = [
# 	Detection("image_0002.jpg", [39, 63, 203, 112], [54, 66, 198, 114]),
# 	Detection("image_0016.jpg", [49, 75, 203, 125], [42, 78, 186, 126]),
# 	Detection("image_0075.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
# 	Detection("image_0090.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
# 	Detection("image_0120.jpg", [35, 51, 196, 110], [36, 60, 180, 108])]
#
# # loop over the example detections
# for detection in examples:
# 	# load the image
# 	image = cv2.imread(detection.image_path)
#
# 	# draw the ground-truth bounding box along with the predicted
# 	# bounding box
# 	cv2.rectangle(image, tuple(detection.gt[:2]),
# 		tuple(detection.gt[2:]), (0, 255, 0), 2)
# 	cv2.rectangle(image, tuple(detection.pred[:2]),
# 		tuple(detection.pred[2:]), (0, 0, 255), 2)
#
# 	# compute the intersection over union and display it
# 	iou = bb_intersection_over_union(detection.gt, detection.pred)
# 	cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# 	print("{}: {:.4f}".format(detection.image_path, iou))
#
# 	# show the output image
# 	cv2.imshow("Image", image)
# 	cv2.waitKey(0)
#
#     # iou > 0.5 is good
