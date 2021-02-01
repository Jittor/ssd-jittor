import jittor as jt
from jittor import nn, init
from utils import *
from math import sqrt
import numpy as np

class L1Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
    
    def execute(self, input, target):
        ret = jt.abs(input - target)
        if self.reduction != None:
            ret = jt.mean(ret) if self.reduction == 'mean' else jt.sum(ret)
        return ret

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def execute(self, input, target):
        bs_idx = jt.array(range(input.shape[0]))
        ret = (- jt.log(nn.softmax(input, dim=1)))[bs_idx, target]
        if self.reduction != None:
            ret = jt.mean(ret) if self.reduction == 'mean' else jt.sum(ret)
        return ret

class VGGBase(nn.Module):
    """ VGG base convolutions to produce lower-level feature maps. """
    def __init__(self):
        super(VGGBase, self).__init__()
        self.conv1_1 = nn.Conv(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.conv2_1 = nn.Conv(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.conv3_1 = nn.Conv(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.Pool(kernel_size=2, stride=2, ceil_mode=True, op='maximum')
        self.conv4_1 = nn.Conv(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.Pool(kernel_size=2, stride=2, op='maximum')
        self.conv5_1 = nn.Conv(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.Pool(kernel_size=3, stride=1, padding=1, op='maximum')
        self.conv6 = nn.Conv(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv(1024, 1024, kernel_size=1)

    def execute(self, image):
        """Forward propagation.
        
        Args:
            image: images, a array of dimensions (N, 3, 300, 300)
        Return: 
            lower-level feature maps conv4_3 and conv7 
        """
        out = nn.relu(self.conv1_1(image))
        out = nn.relu(self.conv1_2(out))
        out = self.pool1(out)
        out = nn.relu(self.conv2_1(out))
        out = nn.relu(self.conv2_2(out))
        out = self.pool2(out)
        out = nn.relu(self.conv3_1(out))
        out = nn.relu(self.conv3_2(out))
        out = nn.relu(self.conv3_3(out))
        out = self.pool3(out)
        out = nn.relu(self.conv4_1(out))
        out = nn.relu(self.conv4_2(out))
        out = nn.relu(self.conv4_3(out))
        conv4_3_feats = out
        out = self.pool4(out)
        out = nn.relu(self.conv5_1(out))
        out = nn.relu(self.conv5_2(out))
        out = nn.relu(self.conv5_3(out))
        out = self.pool5(out)
        out = nn.relu(self.conv6(out))
        conv7_feats = nn.relu(self.conv7(out))

        return (conv4_3_feats, conv7_feats)

class AuxiliaryConvolutions(nn.Module):
    """ Additional convolutions to produce higher-level feature maps. """
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()
        self.conv8_1 = nn.Conv(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv9_1 = nn.Conv(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv10_1 = nn.Conv(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv(128, 256, kernel_size=3, padding=0)
        self.conv11_1 = nn.Conv(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv(128, 256, kernel_size=3, padding=0)
        self.init_conv2d()

    def init_conv2d(self):
        """ Initialize convolution parameters. """
        for c in self.children():
            if isinstance(c, nn.Conv):
                init.gauss_(c.weight)

    def execute(self, conv7_feats):
        """Forward propagation.

        Args:
            conv7_feats: lower-level conv7 feature map, a array of dimensions (N, 1024, 19, 19)
        Return: 
            higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = nn.relu(self.conv8_1(conv7_feats))
        out = nn.relu(self.conv8_2(out))
        conv8_2_feats = out
        out = nn.relu(self.conv9_1(out))
        out = nn.relu(self.conv9_2(out))
        conv9_2_feats = out
        out = nn.relu(self.conv10_1(out))
        out = nn.relu(self.conv10_2(out))
        conv10_2_feats = out
        out = nn.relu(self.conv11_1(out))
        conv11_2_feats = nn.relu(self.conv11_2(out))
        return (conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)

class PredictionConvolutions(nn.Module):
    """Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        Args:
            n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()
        self.n_classes = n_classes
        n_boxes = {
            'conv4_3': 4,
            'conv7': 6,
            'conv8_2': 6,
            'conv9_2': 6,
            'conv10_2': 4,
            'conv11_2': 4,
        }
        self.loc_conv4_3 = nn.Conv(512, (n_boxes['conv4_3'] * 4), kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv(1024, (n_boxes['conv7'] * 4), kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv(512, (n_boxes['conv8_2'] * 4), kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv(256, (n_boxes['conv9_2'] * 4), kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv(256, (n_boxes['conv10_2'] * 4), kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv(256, (n_boxes['conv11_2'] * 4), kernel_size=3, padding=1)
        self.cl_conv4_3 = nn.Conv(512, (n_boxes['conv4_3'] * n_classes), kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv(1024, (n_boxes['conv7'] * n_classes), kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv(512, (n_boxes['conv8_2'] * n_classes), kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv(256, (n_boxes['conv9_2'] * n_classes), kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv(256, (n_boxes['conv10_2'] * n_classes), kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv(256, (n_boxes['conv11_2'] * n_classes), kernel_size=3, padding=1)
        self.init_conv2d()

    def init_conv2d(self):
        """ Initialize convolution parameters. """
        for c in self.children():
            if isinstance(c, nn.Conv):
                init.gauss_(c.weight)

    def execute(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """ Forward propagation.

        Args:
            conv4_3_feats: conv4_3 feature map, a array of dimensions (N, 512, 38, 38)
            conv7_feats: conv7 feature map, a array of dimensions (N, 1024, 19, 19)
            conv8_2_feats: conv8_2 feature map, a array of dimensions (N, 512, 10, 10)
            conv9_2_feats: conv9_2 feature map, a array of dimensions (N, 256, 5, 5)
            conv10_2_feats: conv10_2 feature map, a array of dimensions (N, 256, 3, 3)
            conv11_2_feats: conv11_2 feature map, a array of dimensions (N, 256, 1, 1)
        Return: 
            8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.shape[0]
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = jt.transpose(l_conv4_3, [0, 2, 3, 1])
        l_conv4_3 = jt.reshape(l_conv4_3, [batch_size, -1, 4])
        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = jt.transpose(l_conv7, [0, 2, 3, 1])
        l_conv7 = jt.reshape(l_conv7, [batch_size, -1, 4])
        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        l_conv8_2 = jt.transpose(l_conv8_2, [0, 2, 3, 1])
        l_conv8_2 = jt.reshape(l_conv8_2, [batch_size, -1, 4])
        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        l_conv9_2 = jt.transpose(l_conv9_2, [0, 2, 3, 1])
        l_conv9_2 = jt.reshape(l_conv9_2, [batch_size, -1, 4])
        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)
        l_conv10_2 = jt.transpose(l_conv10_2, [0, 2, 3, 1])
        l_conv10_2 = jt.reshape(l_conv10_2, [batch_size, -1, 4])
        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)
        l_conv11_2 = jt.transpose(l_conv11_2, [0, 2, 3, 1])
        l_conv11_2 = jt.reshape(l_conv11_2, [batch_size, -1, 4])
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = jt.transpose(c_conv4_3, [0, 2, 3, 1])
        c_conv4_3 = jt.reshape(c_conv4_3, [batch_size, -1, self.n_classes])
        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = jt.transpose(c_conv7, [0, 2, 3, 1])
        c_conv7 = jt.reshape(c_conv7, [batch_size, -1, self.n_classes])
        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = jt.transpose(c_conv8_2, [0, 2, 3, 1])
        c_conv8_2 = jt.reshape(c_conv8_2, [batch_size, -1, self.n_classes])
        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = jt.transpose(c_conv9_2, [0, 2, 3, 1])
        c_conv9_2 = jt.reshape(c_conv9_2, [batch_size, -1, self.n_classes])
        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)
        c_conv10_2 = jt.transpose(c_conv10_2, [0, 2, 3, 1])
        c_conv10_2 = jt.reshape(c_conv10_2, [batch_size, -1, self.n_classes])
        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)
        c_conv11_2 = jt.transpose(c_conv11_2, [0, 2, 3, 1])
        c_conv11_2 = jt.reshape(c_conv11_2, [batch_size, -1, self.n_classes])
        locs = jt.contrib.concat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        classes_scores = jt.contrib.concat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)
        return (locs, classes_scores)

class SSD300(nn.Module):
    """ The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions. """

    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        self.n_classes = n_classes
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)
        self.rescale_factors = jt.zeros([1, 512, 1, 1])
        init.constant_(self.rescale_factors, 20)
        self.priors_cxcy = self.create_prior_boxes()

    def execute(self, image):
        """ Forward propagation.
            
        Args:
            image: images, a array of dimensions (N, 3, 300, 300)
        Return: 
            8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        (conv4_3_feats, conv7_feats) = self.base(image)
        norm = conv4_3_feats.sqr().sum(dim=1, keepdims=True).sqrt()
        conv4_3_feats = (conv4_3_feats / norm)
        conv4_3_feats = (conv4_3_feats * self.rescale_factors)
        (conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats) = self.aux_convs(conv7_feats)
        (locs, classes_scores) = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)
        return (locs, classes_scores)

    def create_prior_boxes(self):
        """ Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        Return: prior boxes in center-size coordinates, a array of dimensions (8732, 4)
        """
        fmap_dims = {
            'conv4_3': 38,
            'conv7': 19,
            'conv8_2': 10,
            'conv9_2': 5,
            'conv10_2': 3,
            'conv11_2': 1,
        }
        obj_scales = {
            'conv4_3': 0.1,
            'conv7': 0.2,
            'conv8_2': 0.375,
            'conv9_2': 0.55,
            'conv10_2': 0.725,
            'conv11_2': 0.9,
        }
        aspect_ratios = {
            'conv4_3': [1.0, 2.0, 0.5],
            'conv7': [1.0, 2.0, 3.0, 0.5, 0.333],
            'conv8_2': [1.0, 2.0, 3.0, 0.5, 0.333],
            'conv9_2': [1.0, 2.0, 3.0, 0.5, 0.333],
            'conv10_2': [1.0, 2.0, 0.5],
            'conv11_2': [1.0, 2.0, 0.5],
        }
        fmaps = list(fmap_dims.keys())
        prior_boxes = []
        for (k, fmap) in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = ((j + 0.5) / fmap_dims[fmap])
                    cy = ((i + 0.5) / fmap_dims[fmap])
                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, (obj_scales[fmap] * sqrt(ratio)), (obj_scales[fmap] / sqrt(ratio))])
                        if (ratio == 1.0):
                            try:
                                additional_scale = sqrt((obj_scales[fmap] * obj_scales[fmaps[(k + 1)]]))
                            except IndexError:
                                additional_scale = 1.0
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])
        prior_boxes = np.array(prior_boxes)
        prior_boxes = np.clip(prior_boxes, 0., 1.)
        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """ Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        Args:
            predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
            min_score: minimum threshold for a box to be considered a match for a certain class
            max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
            top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        
        Return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.shape[0]
        n_priors = self.priors_cxcy.shape[0]
        predicted_scores = nn.softmax(predicted_scores, dim=2)
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()
        predicted_locs = predicted_locs.data
        predicted_scores = predicted_scores.data
        assert (n_priors == predicted_locs.shape[1])
        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))
            image_boxes = list()
            image_labels = list()
            image_scores = list()
            for c in range(1, self.n_classes):
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = (class_scores >= min_score)
                n_above_min_score = score_above_min_score.sum()
                if (n_above_min_score == 0):
                    continue
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]
                sort_ind = np.argsort(-class_scores, axis=0)
                class_scores = class_scores[sort_ind]
                class_decoded_locs = class_decoded_locs[sort_ind]
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)
                suppress = np.zeros((n_above_min_score)).astype('int')
                for box in range(class_decoded_locs.shape[0]):
                    if (suppress[box] == 1):
                        continue
                    suppress = np.maximum(suppress, (overlap[box] > max_overlap))
                    suppress[box] = 0
                image_boxes.append(class_decoded_locs[(1-suppress).astype('bool')])
                image_labels.append(int((1-suppress).sum()) * [c])
                image_scores.append(class_scores[(1-suppress).astype('bool')])
            if (len(image_boxes) == 0):
                image_boxes.append(np.array([[0.0, 0.0, 1.0, 1.0]]))
                image_labels.append(np.array([0]))
                image_scores.append(np.array([0.0]))
            image_boxes = np.concatenate(image_boxes, 0)
            image_labels = np.concatenate(image_labels, 0)
            image_scores = np.concatenate(image_scores, 0)
            n_objects = image_scores.shape[0]
            if (n_objects > top_k):
                sort_ind = np.argsort(-image_scores, axis=0)
                image_scores = image_scores[sort_ind][:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_labels[sort_ind][:top_k]
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
        return (all_images_boxes, all_images_labels, all_images_scores)

def argmax(overlap, axis):
    return np.argmax(overlap, axis=axis), np.max(overlap, axis=axis)

class MultiBoxLoss(nn.Module):
    """ The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.smooth_l1 = L1Loss(reduction='sum')
        self.cross_entropy = CrossEntropyLoss(reduce=False, reduction=None)

    def execute(self, predicted_locs, predicted_scores, boxes, labels):
        """ Forward propagation.

        Args:
            predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
            predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
            boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
            labels: true object labels, a list of N tensors
        Return: multibox loss, a scalar
        """
        batch_size = predicted_locs.shape[0]
        n_priors = self.priors_cxcy.shape[0]
        n_classes = predicted_scores.shape[2]
        assert (n_priors == predicted_locs.shape[1] == predicted_scores.shape[1])
        true_locs = np.zeros((batch_size, n_priors, 4))
        true_classes = np.zeros((batch_size, n_priors))
        for i in range(batch_size):
            # Step1: Select one object for every prior
            # Step2: Select one prior for every object, and set its iou 1
            # Step3: Set priors as background, whose iou is lower than threshold, eg: 0.5
            n_objects = boxes[i].shape[0]
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)
            object_for_each_prior, overlap_for_each_prior = argmax(overlap, axis=0)
            prior_for_each_object, _ = argmax(overlap, axis=1)
            object_for_each_prior[prior_for_each_object] = range(n_objects)
            overlap_for_each_prior[prior_for_each_object] = 1.0
            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)
        true_classes = jt.array(true_classes).float32().stop_grad()
        true_locs = jt.array(true_locs).float32().stop_grad()
        positive_priors = (true_classes != 0)
        loc_loss = self.smooth_l1(
           (predicted_locs * positive_priors.broadcast([1,1,4], [2])),  
           (true_locs * positive_priors.broadcast([1,1,4], [2]))
        )
        loc_loss /= (positive_priors.float32().sum() * 4)
        n_positives = positive_priors.float32().sum(1)
        n_hard_negatives = self.neg_pos_ratio * n_positives
        conf_loss_all = self.cross_entropy(
            jt.reshape(predicted_scores, [-1, n_classes]), jt.reshape(true_classes, [-1,])
        )
        conf_loss_all = jt.reshape(conf_loss_all, [batch_size, n_priors])
        conf_loss_pos = conf_loss_all * positive_priors
        conf_loss_neg = conf_loss_all * (1 - positive_priors)
        _, conf_loss_neg = conf_loss_neg.argsort(dim=1, descending=True)
        hardness_ranks = jt.array(range(n_priors)).broadcast([conf_loss_neg.shape[0], conf_loss_neg.shape[1]], [0])
        hard_negatives = hardness_ranks < n_hard_negatives.broadcast(hardness_ranks.shape, [1])
        conf_loss_hard_neg = conf_loss_neg * hard_negatives
        conf_loss = ((conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.float32().sum())
        return (conf_loss + (self.alpha * loc_loss)), conf_loss, loc_loss