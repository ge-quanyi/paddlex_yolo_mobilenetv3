# paddlex_yolo_mobilenetv3
本地yolo_mobilenetv3训练



**本项目为利用paddlex工具训练yolo_mobilenetv3模型，并进行剪裁操作，最终可转化成.nb模型在树莓派上部署**

脚本说明：

```
train.py  模型训练   
params_analysis.py 参数评估，模型剪裁预备操作   
slim_visualize.py 可视化操作   
prune_train.py 模型剪裁    
paddlex --export_inference --model_dir=./output/yolov3_mobilenetv3/best_model --save_dir=./inference_model  #导出  inference 模型   
opt.py   转化为.nb模型
```

