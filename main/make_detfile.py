import pickle

def file_read():
    with open('cache/predictions.pkl', 'rb') as read:
        predictions = pickle.load(read)

    with open('../data/VOC2007/ImageSets/Layout/val.txt', 'r') as read:
        lines = read.readlines()
    splitlines = [x.strip().split(' ') for x in lines]

    return predictions, splitlines

def file_maker(predictions, imagenames):
    voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                   'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                   'Sheep', 'Sofa', 'Train', 'Tvmonitor']

    for i, classes in enumerate(voc_classes):
        with open('../data/predictions/{}.txt'.format(classes.lower()), 'w') as w:
            for j, prediction in enumerate(predictions):
                for k, l in enumerate(prediction):
                    if prediction[k][0] == classes:
                        w.write('{} {} {} {} {} {}\n'.format(imagenames[j][0], l[1], l[2], l[3], l[4], l[5]))
                    else:
                        pass

if __name__ == '__main__':
    predictions, imagenames = file_read()

    file_maker(predictions, imagenames)
