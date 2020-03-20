import time
import jittor as jt
from jittor import nn, Module
from utils import *
from model import SSD300, MultiBoxLoss
import pickle
import random
import numpy as np
from datasets import PascalVOCDataset
jt.flags.use_cuda = 1   

def setseed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

from tensorboardX import SummaryWriter
exp_id = "7"
writer = SummaryWriter('tensorboard/' + exp_id) 

# Data parameters
data_folder = '/home/storage/zwy/jitcode/object_detection/ssd/dataset/'
keep_difficult = True # 是否保留那些比较难检测的物体
n_classes = len(label_map)

# Learning parameters
batch_size = 20   # batch大小
iterations = 120000  # 一共要训的轮数
decay_lr_at = [80000, 100000]  # 在这些轮的时候学习率乘以0.1
start_epoch = 0  # 开始epoch
print_freq = 1  # train的时候，多少个iter打印一次信息
lr = 1e-3  # 学习率
momentum = 0.9  # SGD的momentum
weight_decay = 5e-4  # SGD的weight_decay
grad_clip = 1  # 设置是否要把梯度clamp到[-grad_clip, grad_clip]，如果是None则不clip
best_mAP = 0. # 记录当前最高mAP
train_loader = PascalVOCDataset(data_folder,
                                    split='train',
                                    keep_difficult=keep_difficult, batch_size=batch_size, shuffle=False, data_argu=True)
length = len(train_loader) // batch_size # 一个batch的iters
epochs = iterations // (len(train_loader) // 32) # 原论文batch_size为32跑了120000个iters，由此计算出epoches
decay_lr_at = [it // (len(train_loader) // 32) for it in decay_lr_at] # 并计算出需要降lr的epochs集合
val_loader = PascalVOCDataset(data_folder,
                                split='test',   
                                keep_difficult=keep_difficult, batch_size=batch_size, shuffle=False)
                                            
model = SSD300(n_classes=n_classes)
optimizer = nn.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
model.load_parameters(pickle.load(open("init.pkl", "rb")))
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)

setseed(19961107)

def main():
    for epoch in range(start_epoch, epochs):
        if epoch in decay_lr_at:
            optimizer.lr *= 0.1
        train(train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch)
        writer.add_scalar('Train/lr', optimizer.lr, global_step=epoch)
        if epoch % 5 == 0 and epoch > 0:
            evaluate(test_loader=val_loader, model=model)


def train(train_loader, model, criterion, optimizer, epoch):
    global best_loss, exp_id
    model.train()

    for i, (images, boxes, labels, _) in enumerate(train_loader):
        images = jt.array(images)

        predicted_locs, predicted_scores = model(images)
        loss, conf_loss, loc_loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        if grad_clip is not None:
            optimizer.grad_clip = grad_clip
        optimizer.step(loss)
        
        if i % print_freq == 0:
            print(f'[Train] Experiment id: {exp_id} || Epochs: [{epoch}/{epochs}] || Iters: [{i}/{length}] || Loss: {loss} || Best mAP: {best_mAP}')
            writer.add_scalar('Train/Loss', loss.data[0], global_step=i + epoch * length)
            writer.add_scalar('Train/Loss_conf', conf_loss.data[0], global_step=i + epoch * length)
            writer.add_scalar('Train/Loss_loc', loc_loss.data[0], global_step=i + epoch * length)

cnt_val = 0
def evaluate(test_loader, model):
    global best_mAP, cnt_val
    model.eval()

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    for i, (images, boxes, labels, difficulties) in enumerate(test_loader):
        print(f'[Test] Evaluate Iters: [{i}/{length}]')
        images = jt.array(images)
        predicted_locs, predicted_scores = model(images)

        det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2, max_overlap=0.45, top_k=200)

        det_boxes.extend(det_boxes_batch)
        det_labels.extend(det_labels_batch)
        det_scores.extend(det_scores_batch)
        true_boxes.extend(boxes)
        true_labels.extend(labels)
        true_difficulties.extend(difficulties)

    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    writer.add_scalar('Test/mAP', mAP, global_step=cnt_val)
    cnt_val += 1
    print(f'[*] Mean Average Precision (mAP): {mAP}, best_mAP: {best_mAP}')
    if mAP > best_mAP:
        best_mAP = mAP
        print(f'[*] Update best_mAP to {best_mAP}')
        model.save(os.path.join("tensorboard", exp_id, 'model_best.pkl'))
    model.save(os.path.join("tensorboard", exp_id, 'model_last.pkl'))


if __name__ == '__main__':
    main()
