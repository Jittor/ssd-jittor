from jittor.dataset.dataset import Dataset
import json
import os
import cv2
import numpy as np
from utils import random_crop, random_bright, random_swap, random_contrast, random_saturation, random_hue, random_flip, random_expand
import random
from pdb import set_trace as st

class PascalVOCDataset(Dataset):
    def __init__(self, data_folder, split, keep_difficult=False, batch_size=1, shuffle=False, data_argu=False):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = True
        self.data_argu = data_argu

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)
        self.total_len = len(self.images)
        print(f"[*] Loading {self.split} {self.total_len} images.")
        self.set_attrs(batch_size = self.batch_size, total_len = self.total_len, shuffle = self.shuffle, drop_last = self.drop_last)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype("float32")
        objects = self.objects[i]
        boxes = np.array(objects['boxes']).astype("float32")
        labels = np.array(objects['labels'])
        difficulties = np.array(objects['difficulties'])

        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]
        if self.split == 'TRAIN' and self.data_argu:
            data_enhance = [random_bright, random_contrast, random_saturation, random_hue]
            random.shuffle(data_enhance)
            for d in data_enhance:
                image = d(image)
            if random.random() < 0.5:
                image, boxes = random_expand(image, boxes)
            image, boxes, labels, difficulties = random_crop(image, boxes, labels, difficulties)
            image, boxes = random_flip(image, boxes)
        height, width, _ = image.shape
        image = cv2.resize(image, (300, 300))
        image /= 255.
        image = (image - self.mean) / self.std
        image = image.transpose((2,0,1)).astype("float32")
        
        boxes[:,[0,2]] /= width
        boxes[:,[1,3]] /= height
        
        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_batch(self, batch):
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = np.stack(images, axis=0)
        return images, boxes, labels, difficulties