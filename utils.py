import json
import os
import jittor as jt
import xml.etree.ElementTree as ET
import numpy as np
import random
import cv2

# Label map
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def parse_annotation(annotation_path):
    """Parse annotation file

    Args:
        annotation_path(str): Annotation file path
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, voc12_path, output_folder):
    """Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    Args:
        voc07_path(str): path to the 'VOC2007' folder
        voc12_path(str): path to the 'VOC2012' folder
        output_folder(str): folder where the JSONs must be saved
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in [voc07_path, voc12_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Validation data
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in validation data
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


def xy_to_cxcy(xy):
    """Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    Args:
        xy(numpy.array): bounding boxes in boundary coordinates, a array of size (n_boxes, 4)
    Return: 
        (numpy.array): bounding boxes in center-size coordinates, a array of size (n_boxes, 4)
    """
    return np.concatenate(((xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]), 1)  # w, h


def cxcy_to_xy(cxcy):
    """Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    Args:
        cxcy(numpy.array): bounding boxes in center-size coordinates, a array of size (n_boxes, 4)
    Return:
        (numpy.array): bounding boxes in boundary coordinates, a array of size (n_boxes, 4)
    """
    return np.concatenate((cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)), 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    Args:
        cxcy(numpy.array): bounding boxes in center-size coordinates, a array of size (n_priors, 4)
        priors_cxcy(numpy.array): prior boxes with respect to which the encoding must be performed, a array of size (n_priors, 4)
    Return: 
        (numpy.array): encoded bounding boxes, a array of size (n_priors, 4)
    """
    return np.concatenate(((cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      np.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5), 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    Args:
        gcxgcy(numpy.array): encoded bounding boxes, i.e. output of the model, a array of size (n_priors, 4)
        priors_cxcy(numpy.array): prior boxes with respect to which the encoding is defined, a array of size (n_priors, 4)
    Return: 
        (numpy.array): decoded bounding boxes in center-size form, a array of size (n_priors, 4)
    """
    return np.concatenate((gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y 
        np.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]), 1)  # w, h


def find_intersection(set_1, set_2):
    """Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    Args:
        set_1(numpy.array): set 1, a array of dimensions (n1, 4)
        set_2(numpy.array): set 2, a array of dimensions (n2, 4)
    Return: 
        (numpy.array): intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a array of dimensions (n1, n2)
    """
    lower_bounds = np.maximum(np.expand_dims(set_1[:, :2], 1), np.expand_dims(set_2[:, :2], 0))  # (n1, n2, 2)
    upper_bounds = np.minimum(np.expand_dims(set_1[:, 2:], 1), np.expand_dims(set_2[:, 2:], 0))  # (n1, n2, 2)
    intersection_dims = np.maximum(upper_bounds - lower_bounds, 0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    Args:
        set_1(numpy.array): set 1, a array of dimensions (n1, 4)
        set_2(numpy.array): set 2, a array of dimensions (n2, 4)
    Return: 
        (numpy.array): Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a array of dimensions (n1, n2)
    """
    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union.
    union = np.expand_dims(areas_set_1, 1) + np.expand_dims(areas_set_2, 0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

# Data Augmentation 
def random_bright(im, delta=32):
    """Random change bright
    
    Args:
        im(numpy.array): one H,W,3 image
        delta(int): bright ratio
    """
    if random.random() < 0.5:
        delta = random.uniform(-delta, delta)
        im += delta
        im = im.clip(min=0, max=255)
    return im

def random_swap(im):
    """Random swap channel
    
    Args:
        im(numpy.array): one H,W,3 image
    """
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    if random.random() < 0.5:
        swap = perms[random.randrange(0, len(perms))]
        im = im[:, :, swap]
    return im

def random_contrast(im, lower=0.5, upper=1.5):
    """Random change contrast
    
    Args:
        im(numpy.array): one H,W,3 image
        lower(float): change lower limit 
        upper(float): change upper limit
    """
    if random.random() < 0.5:
        alpha = random.uniform(lower, upper)
        im *= alpha
        im = im.clip(min=0, max=255)
    return im

def random_saturation(im, lower=0.5, upper=1.5):
    """Random change saturation
    
    Args:
        im(numpy.array): one H,W,3 image
        lower(float): change lower limit 
        upper(float): change upper limit
    """
    if random.random() < 0.5:
        im[:, :, 1] *= random.uniform(lower, upper)
    return im

def random_hue(im, delta=18.0):
    """Random change hue
    
    Args:
        im(numpy.array): one H,W,3 image
        delta(int): hue ratio
    """
    if random.random() < 0.5:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        im[:, :, 0] += random.uniform(-delta, delta)
        im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
        im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
        im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
    return im

def random_flip(image, boxes):
    """Random flip the image and its boxes
    
    Args:
        image(numpy.array): one H,W,3 image
        boxes(numpy.array): its ground truth bounding boxes
    """
    if random.random() < 0.5:
        new_image = cv2.flip(image, 1)
        H, W, C = new_image.shape
        new_boxes = boxes
        new_boxes[:, 0] = W - boxes[:, 0] - 1
        new_boxes[:, 2] = W - boxes[:, 2] - 1
        new_boxes = new_boxes[:, [2, 1, 0, 3]]
        return new_image, new_boxes
    return image, boxes

def random_expand(image, boxes):
    """
    Perform a zooming out operation by placing the image.

    Helps to learn to detect smaller objects.

    Args:
        image(numpy.array): image, a array of dimensions (original_h, original_w, 3)
        boxes(numpy.array): bounding boxes in boundary coordinates, a array of dimensions (n_objects, 4)
    :return: expanded image, updated bounding box coordinates
    """
    original_h, original_w, C = image.shape
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)
    new_image = np.random.rand(new_h, new_w, 3) * 255.

    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[top:bottom, left:right, :] = image

    new_boxes = boxes + np.array([[left, top, left, top]])
    return new_image, new_boxes

def random_crop(image, boxes, labels, difficulties):
    """Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    Args:
        image(numpy.array): image, a array of dimensions (original_h, original_w, 3)
        boxes(numpy.array): bounding boxes in boundary coordinates, a array of dimensions (n_objects, 4)
        labels(numpy.array): labels of objects, a array of dimensions (n_objects)
        difficulties(numpy.array): difficulties of detection of these objects, a array of dimensions (n_objects)
    Return:
        cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h, original_w, _ = image.shape
    while True:
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])
        if min_overlap is None:
            return image, boxes, labels, difficulties
        max_trials = 50
        for _ in range(max_trials):
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = np.array([left, top, right, bottom])
            overlap = find_jaccard_overlap(crop[np.newaxis,:], boxes)[0]

            if overlap.max() < min_overlap:
                continue
            
            new_image = image[top:bottom, left:right, :]
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (bb_centers[:, 1] < bottom)

            if not centers_in_crop.any():
                continue

            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            new_boxes[:, :2] = np.maximum(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = np.minimum(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties

# Evaluation
def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    Args:
        det_boxes: list of arrays, one array for each image containing detected objects' bounding boxes
        det_labels: list of arrays, one array for each image containing detected objects' labels
        det_scores: list of arrays, one array for each image containing detected objects' labels' scores
        true_boxes: list of arrays, one array for each image containing actual objects' bounding boxes
        true_labels: list of arrays, one array for each image containing actual objects' labels
        true_difficulties: list of arrays, one array for each image containing actual objects' difficulty (0 or 1)
    Return: 
        list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)
    
    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].shape[0])
    true_images = np.array(true_images)
    true_boxes = np.concatenate(true_boxes, 0)  # (n_objects, 4)
    true_labels = np.concatenate(true_labels, 0)  # (n_objects)
    true_difficulties = np.concatenate(true_difficulties, 0)  # (n_objects)

    assert true_images.shape[0] == true_boxes.shape[0] == true_labels.shape[0]

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].shape[0])
    det_images = np.array(det_images)
    det_boxes = np.concatenate(det_boxes, 0)  # (n_detections, 4)
    det_labels = np.concatenate(det_labels, 0)  # (n_detections)
    det_scores = np.concatenate(det_scores, 0)  # (n_detections)

    assert det_images.shape[0] == det_boxes.shape[0] == det_labels.shape[0] == det_scores.shape[0]
    # Calculate APs for each class (except background)
    average_precisions = np.zeros(n_classes - 1)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = np.zeros(true_class_difficulties.shape[0])  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.shape[0]
        if n_class_detections == 0:
            continue
        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = np.sort(-det_class_scores), np.argsort(-det_class_scores)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = np.zeros(n_class_detections)  # (n_class_detections)
        false_positives = np.zeros(n_class_detections)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d][np.newaxis, :]  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.shape[0] == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = np.max(overlaps[0]), np.argmax(overlaps[0])  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = np.array(range(true_class_boxes.shape[0]))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1
        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = np.cumsum(true_positives, 0)  # (n_class_detections)
        cumul_false_positives = np.cumsum(false_positives, 0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = np.arange(0, 1.1, .1).tolist()  # (11)
        precisions = np.zeros(len(recall_thresholds))  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]
    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean()
    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision
