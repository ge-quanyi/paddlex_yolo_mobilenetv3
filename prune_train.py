import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx

# insect_dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
# pdx.utils.download_and_decompress(insect_dataset, path='./')

train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(
        target_size=320, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=320, interp='CUBIC'),
    transforms.Normalize(),
])

train_dataset = pdx.datasets.VOCDetection(
    data_dir='MyDataset',
    file_list='MyDataset/train_list.txt',
    label_list='MyDataset/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='MyDataset',
    file_list='MyDataset/val_list.txt',
    label_list='MyDataset/labels.txt',
    transforms=eval_transforms)

num_classes = len(train_dataset.labels)

model = pdx.det.YOLOv3(num_classes=num_classes, backbone='MobileNetV3_large')

model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.0000625,
    lr_decay_epochs=[210, 240],
    pretrain_weights='output/yolov3_mobilenetv3/best_model',
    save_dir='output/yolov3_mobilenetv3_prune',
    sensitivities_file='./yolov3.sensi.data',
    eval_metric_loss=0.05,
    use_vdl=True,
    early_stop=True,
    early_stop_patience=5)