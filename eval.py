import jittor as jt
from utils import calculate_mAP
from datasets import PascalVOCDataset
from model import SSD300
from pprint import PrettyPrinter
import pickle
import os

jt.flags.use_cuda = 1

# Parameters
data_folder = 'dataset/'
keep_difficult = True
batch_size = 47
pp = PrettyPrinter()

# Define model & Load parameters
experiment_id = "pretrain_model"
model_path = os.path.join('tensorboard', experiment_id, 'model_best.pkl')
params = pickle.load(open(model_path, "rb"))
model = SSD300(21)
model.load_parameters(params)
print(f'[*] Load model {model_path} success')

# Load test data
test_loader = PascalVOCDataset(data_folder,
                              split='test',   
                              keep_difficult=keep_difficult, batch_size=batch_size, shuffle=False)
length = len(test_loader) // batch_size

def evaluate(test_loader, model):
    """ Evaluate.
    Args:
        test_loader: DataLoader for test data
        model: model
    """
    model.eval()

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    for i, (images, boxes, labels, difficulties) in enumerate(test_loader):
        print(f'Iters: [{i}/{length}]')
        images = jt.array(images)
        predicted_locs, predicted_scores = model(images)

        det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores, min_score=0.01, max_overlap=0.45, top_k=200)

        det_boxes.extend(det_boxes_batch)
        det_labels.extend(det_labels_batch)
        det_scores.extend(det_scores_batch)
        true_boxes.extend(boxes)
        true_labels.extend(labels)
        true_difficulties.extend(difficulties)

    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)

if __name__ == '__main__':
    evaluate(test_loader, model)