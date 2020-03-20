这是Jittor的目标检测SSD教程，参考自[sgrvinod](<https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection>)。

# 模型训练

1.下载VOC数据集，一共三个，地址

* http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

* http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

* http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

并将它们解压到data/文件夹，形成data/VOCdevkit/VOC2007/和data/VOCdevkit/VOC2012/结构。

2.执行python3.7 create_data_lists.py，会在dataset生成5个json文件。

3.从[链接](https://cloud.tsinghua.edu.cn/f/50e24573693e4cd7b737/?dl=1)下载VGG16初始化模型init.pkl，放在根目录下。

4.训练命令：python3.7 train.py

注：如果要进行多个实验，请自行修改exp_id。若想在tensorboard里查看,请运行tensorboard --logdir tensorboard/

# 模型测试

## 预训练模型

如果您想用我们训练好的model进行测试，请在[链接](https://cloud.tsinghua.edu.cn/f/5021408263134ed5b53d/?dl=1)下载model_best.pkl，并将其放在tensorborad/pretrain_model/model_best.pkl。

## 测试mAP

命令：python3.7 eval.py

注：请自行修改exp_id为您训练好的model。如果下载了pretrain_model，请将其放置在tensorborad/pretrain_model/model_best.pkl，并将eval.py的exp_id改为pretrain_model。

## 测试图片

命令：python3.7 detect.py

注：请自行修改exp_id为您训练好的model。



