import jittor as jt
from PIL import ImageDraw, ImageFont
import numpy as np
import pickle
from PIL import Image
import cv2
import time
from model import SSD300
from utils import *
import os
jt.flags.use_cuda = 1

# Load model checkpoint
experiment_id = "pretrain_model" # set your experiment id
model_path = os.path.join('tensorboard', experiment_id, 'model_best.pkl')
params = pickle.load(open(model_path, "rb"))
model = SSD300(21)
model.load_parameters(params)
print(f'[*] Load model {model_path} success')

# Transforms
def transform(image, size=300, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = cv2.resize(image, (300, 300))
    image /= 255.
    image = (image - mean) / std
    return image.transpose(2,0,1)

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """ Detect objects in an image with a trained SSD300, and visualize the results.
    Args:
        original_image: image, a PIL Image
        min_score: minimum threshold for a detected box to be considered a match for a certain class
        max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
        top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    Return: annotated image, a PIL Image
    """
    image = np.array(original_image).astype('float32')
    H, W, C = image.shape
    image = transform(image)
    image = jt.array(image[np.newaxis,:]).float32()
    predicted_locs, predicted_scores = model(image)
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k)
    det_boxes = det_boxes[0]
    original_dims = np.array([[W, H, W, H]])
    det_boxes = det_boxes * original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0]]
    if det_labels == ['background']:
        return original_image
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("ahronbd.ttf", 15)
    for i in range(det_boxes.shape[0]):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]]) 
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
    del draw
    return annotated_image


if __name__ == '__main__':
    img_dir = 'sample_images'
    try:
        os.makedirs('result/')
    except:
        print('Destination dir exists')
        pass
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        sta = time.time()
        ret = detect(original_image, min_score=0.6, max_overlap=0.3, top_k=20)
        print(f"Once detect cost time: {time.time() - sta}")
        ret.save(os.path.join('result', img_name))