import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

model = pdx.load_model('output/yolov3_mobilenetv3/best_model')

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='MyDataset',
    file_list='MyDataset/val_list.txt',
    label_list='MyDataset/labels.txt',
    transforms=model.eval_transforms)

pdx.slim.prune.analysis(
    model, dataset=eval_dataset, batch_size=4, save_file='yolov3.sensi.data')