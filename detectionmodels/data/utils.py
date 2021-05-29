import numpy as np


def resize_anchors(base_anchors, target_shape, base_shape=(416, 416)):
    '''
    original anchor size is clustered from COCO dataset
    under input shape (416,416). We need to resize it to
    our train input shape for better performance
    '''
    return np.around(base_anchors*target_shape[::-1]/base_shape[::-1])


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()  # string of ints
    anchors = [float(x) for x in anchors.split(',')]  # list of ints
    return np.array(anchors).reshape(-1, 2)  # pair up ints (num_anchors, 2)
