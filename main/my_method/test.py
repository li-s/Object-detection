import ml_metrics as metrics
import xml.etree.ElementTree as ET

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

def bb_intersection_over_union(boxA, boxB):
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
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def get_ground_truth(image_file):
    with open(image_file, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # Gets name of each image file to validate on
    images = []
    for image in imagenames:
        images.append(image + '.txt')

    # Gets ground truth boxes
    ground_truth = []
    for image in imagenames:
        filename = '../../data/VOC2007/Annotations/' + image + '.xml'
        objects = parse_rec(filename)
        for i in objects:
            ground_truth.append(i['name'])
            ground_truth.append(i['bbox'])

    ground_truth = tuple(ground_truth)

    return images, ground_truth

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

def validates_images(image_file, model, bbox_util):
    inputs = []
    images = []
    with open(image_file, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # Gets name of each image file to validate on
    images = []
    for image in imagenames:
        images.append(image + '.txt')

    for filename in sorted(files):
        img_path = '../../data/VOC2007/JPEGImages/' + filename
        img = image.load_img(img_path, target_size=(300, 300))
        img = image.img_to_array(img)
        images.append(imread(img_path))
        inputs.append(img.copy())

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


if __name__ == '__main__':
    images, ground_truth = get_ground_truth('../../data/VOC2007/ImageSets/Layout/val.txt')

    # Formats data
    truth = []
    gt = []
    for i, j in enumerate(ground_truth):
        if i%2 == 0:
            truth.append(j)
        else:
            gt.append(j)

    # Load model
    model, bbox_util = load_model()
    # Performs predicitons
    results, img = validates_images(model, bbox_util)
    # Removes unecessary predicitons
    image_list = process_images(results)




    iou = bb_intersection_over_union(gt, prediction)

    ap = metrics.mapk(truth, prediction, 328) #may need to minus 1
    print('IoU: {}\nAP: {}'.format(iou, ap))








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
