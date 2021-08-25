import paddlex as pdx
model = pdx.load_model('output/yolov3_mobilenetv3_prune/best_model')
# 加载数据集用于量化
dataset = pdx.datasets.ImageNet(
                data_dir='MyDataset',
                file_list='MyDataset/test_list.txt',
                label_list='MyDataset/labels.txt',
                transforms=model.test_transforms)

# 开始量化
pdx.slim.export_quant_model(model, dataset, batch_size=4, batch_num=5, save_dir='./quant_result', cache_dir='./tmp')
