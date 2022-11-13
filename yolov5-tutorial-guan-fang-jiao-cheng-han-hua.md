# YOLOv5 Tutorial 官方教程 汉化\_

该笔记本是由Ultralytics LLC编写的，可根据以下内容免费进行重新分发：[GPL-3.0 license](https://choosealicense.com/licenses/gpl-3.0/).\
欲了解更多信息，请访问 [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) 和 [https://www.ultralytics.com](https://www.ultralytics.com/).

原文：[Google Colab - YOLOv5 Tutorial](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb)

### 设置 <a href="#t0" id="t0"></a>

克隆存储库，安装依赖项并检查[PyTorch](https://so.csdn.net/so/search?q=PyTorch\&spm=1001.2101.3001.7020)和GPU。

```
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt  # install dependencies

import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
123456789
```

```
Setup complete. Using torch 1.7.0+cu101 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', major=7, minor=0, total_memory=16160MB, multi_processor_count=80)
1
```

### 1. 推断 <a href="#t1" id="t1"></a>

`detect.py` 在各种来源上进行推断，然后自动从模型中下载模型 [最新的 YOLOv5 发行版本](https://github.com/ultralytics/yolov5/releases).

```
!python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/
Image(filename='runs/detect/exp/zidane.jpg', width=600)
12
```

```
Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', exist_ok=False, img_size=640, iou_thres=0.45, name='exp', project='runs/detect', save_conf=False, save_txt=False, source='data/images/', update=False, view_img=False, weights=['yolov5s.pt'])
YOLOv5 v4.0-21-gb26a2f6 torch 1.7.0+cu101 CUDA:0 (Tesla V100-SXM2-16GB, 16130.5MB)

Fusing layers... 
Model Summary: 224 layers, 7266973 parameters, 0 gradients, 17.0 GFLOPS
image 1/2 /content/yolov5/data/images/bus.jpg: 640x480 4 persons, 1 buss, 1 skateboards, Done. (0.011s)
image 2/2 /content/yolov5/data/images/zidane.jpg: 384x640 2 persons, 2 ties, Done. (0.011s)
Results saved to runs/detect/exp
Done. (0.110s)
123456789
```

结果保存到`runs/detect`。可用推论来源的完整列表：\


### 2. 测试 <a href="#t2" id="t2"></a>

在[COCO](https://cocodataset.org/#home)上测试模型val或test-dev数据集来评估训练的准确性。\
模型是从[latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases)自动下载的 。分类显示结果，请使用`--verbose` 标示. 注意`pycocotools`指标可能比相同储存库的指标好1-2％, 如下所示，由于mAP计算中的细微差异。

### COCO val2017 <a href="#t3" id="t3"></a>

下载[COCO val 2017](https://github.com/ultralytics/yolov5/blob/74b34872fdf41941cddcf243951cdb090fbac17b/data/coco.yaml#L14)数据集(1GB - 5000 图片),并测试模型的准确性。

```
# Download COCO val2017
torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017val.zip', 'tmp.zip')
!unzip -q tmp.zip -d ../ && rm tmp.zip
123
```

```
HBox(children=(FloatProgress(value=0.0, max=819257867.0), HTML(value='')))
1
```

```
# Run YOLOv5x on COCO val2017
!python test.py --weights yolov5x.pt --data coco.yaml --img 640 --iou 0.65
12
```

```
Namespace(augment=False, batch_size=32, conf_thres=0.001, data='./data/coco.yaml', device='', exist_ok=False, img_size=640, iou_thres=0.65, name='exp', project='runs/test', save_conf=False, save_hybrid=False, save_json=True, save_txt=False, single_cls=False, task='val', verbose=False, weights=['yolov5x.pt'])
YOLOv5 v4.0-75-gbdd88e1 torch 1.7.0+cu101 CUDA:0 (Tesla V100-SXM2-16GB, 16160.5MB)

Downloading https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5x.pt to yolov5x.pt...
100% 168M/168M [00:04<00:00, 39.7MB/s]

Fusing layers... 
Model Summary: 476 layers, 87730285 parameters, 0 gradients, 218.8 GFLOPS
[34m[1mval: [0mScanning '../coco/val2017' for images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100% 5000/5000 [00:01<00:00, 2824.78it/s]
[34m[1mval: [0mNew cache created: ../coco/val2017.cache
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100% 157/157 [01:33<00:00,  1.68it/s]
                 all       5e+03    3.63e+04       0.749       0.619        0.68       0.486
Speed: 5.2/2.0/7.3 ms inference/NMS/total per 640x640 image at batch-size 32

Evaluating pycocotools mAP... saving runs/test/exp/yolov5x_predictions.json...
loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=4.47s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=94.87s).
Accumulating evaluation results...
DONE (t=15.96s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.501
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.687
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.544
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.548
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.637
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.729
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.826
Results saved to runs/test/exp
1234567891011121314151617181920212223242526272829303132333435363738394041
```

### COCO test-dev2017 <a href="#t4" id="t4"></a>

Download [COCO test2017](https://github.com/ultralytics/yolov5/blob/74b34872fdf41941cddcf243951cdb090fbac17b/data/coco.yaml#L15) dataset (7GB - 40,000图片),在测试开发集上测试模型准确性（20,000张图片），结果保存到`*.json` 可以提交给评估服务器的文件https://competitions.codalab.org/competitions/20794.

```
# 下载 COCO test-dev2017
torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip', 'tmp.zip')
!unzip -q tmp.zip -d ../ && rm tmp.zip  # unzip labels
!f="test2017.zip" && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f && rm $f  # 7GB,  41k images
%mv ./test2017 ./coco/images && mv ./coco ../  # move images to /coco and move /coco next to /yolov5
12345
```

```
# Run YOLOv5s on COCO test-dev2017 using --task test
!python test.py --weights yolov5s.pt --data coco.yaml --task test
12
```

### 3. 训练 <a href="#t5" id="t5"></a>

下载 [COCO128](https://www.kaggle.com/ultralytics/coco128), 一个小的 128-image 教程数据集, 启动tensorboard并从预先训练的checkpoint训练YOLOv5s重复 3 epochs (请注意，实际训练通常会更长, 大约 **300-1000 epochs**, 取决于您的数据集).

```
# 下载 COCO128
torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip', 'tmp.zip')
!unzip -q tmp.zip -d ../ && rm tmp.zip
123
```

```
HBox(children=(FloatProgress(value=0.0, max=22091032.0), HTML(value='')))
1
```

训练一个YOLOv5s模型可以使用[COCO128](https://www.kaggle.com/ultralytics/coco128)、`--data coco128.yaml`和预训练的`--weights yolov5s.pt`开始，或从随机初始化的`--weights '' --cfg yolov5s.yaml`开始. 模型是从[latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases)自动下载的, 并且首次使用 **COCO, COCO128, 和 VOC 数据集也是自动下载的** 。

所有训练结果均以递增的运行目录保存到`runs/train/`，也就是说`runs/train/exp2`, `runs/train/exp3`等等。

```
# Tensorboard (可选择的)
%load_ext tensorboard
%tensorboard --logdir runs/train
123
```

```
# Weights & Biases (可选择的)
%pip install -q wandb  
!wandb login  # use 'wandb disabled' or 'wandb enabled' to disable or enable
123
```

```
# 训练 YOLOv5s on COCO128 for 3 epochs
!python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --nosave --cache
12
```

```
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 v4.0-75-gbdd88e1 torch 1.7.0+cu101 CUDA:0 (Tesla V100-SXM2-16GB, 16160.5MB)

Namespace(adam=False, batch_size=16, bucket='', cache_images=True, cfg='', data='./data/coco128.yaml', device='', epochs=3, evolve=False, exist_ok=False, global_rank=-1, hyp='data/hyp.scratch.yaml', image_weights=False, img_size=[640, 640], linear_lr=False, local_rank=-1, log_artifacts=False, log_imgs=16, multi_scale=False, name='exp', noautoanchor=False, nosave=True, notest=False, project='runs/train', quad=False, rect=False, resume=False, save_dir='runs/train/exp', single_cls=False, sync_bn=False, total_batch_size=16, weights='yolov5s.pt', workers=8, world_size=1)
wandb: Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)
Start Tensorboard with "tensorboard --logdir runs/train", view at http://localhost:6006/
2021-02-12 06:38:28.027271: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.1
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
Downloading https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt to yolov5s.pt...
100% 14.1M/14.1M [00:01<00:00, 13.2MB/s]


                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Focus                     [3, 32, 3]                    
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  1    156928  models.common.C3                        [128, 128, 3]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  1    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1    656896  models.common.SPP                       [512, 512, [5, 9, 13]]        
  9                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1    229245  models.yolo.Detect                      [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 283 layers, 7276605 parameters, 7276605 gradients, 17.1 GFLOPS

Transferred 362/362 items from yolov5s.pt
Scaled weight_decay = 0.0005
Optimizer groups: 62 .bias, 62 conv.weight, 59 other
train: Scanning '../coco128/labels/train2017' for images and labels... 128 found, 0 missing, 2 empty, 0 corrupted: 100% 128/128 [00:00<00:00, 2566.00it/s]
train: New cache created: ../coco128/labels/train2017.cache
train: Caching images (0.1GB): 100% 128/128 [00:00<00:00, 175.07it/s]
val: Scanning '../coco128/labels/train2017.cache' for images and labels... 128 found, 0 missing, 2 empty, 0 corrupted: 100% 128/128 [00:00<00:00, 764773.38it/s]
val: Caching images (0.1GB): 100% 128/128 [00:00<00:00, 128.17it/s]
Plotting labels... 

autoanchor: Analyzing anchors... anchors/target = 4.26, Best Possible Recall (BPR) = 0.9946
Image sizes 640 train, 640 test
Using 2 dataloader workers
Logging results to runs/train/exp
Starting training for 3 epochs...

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
       0/2     3.27G   0.04357   0.06781   0.01869    0.1301       207       640: 100% 8/8 [00:03<00:00,  2.03it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100% 4/4 [00:04<00:00,  1.14s/it]
                 all         128         929       0.646       0.627       0.659       0.431

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
       1/2     7.75G   0.04308   0.06654   0.02083    0.1304       227       640: 100% 8/8 [00:01<00:00,  4.11it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100% 4/4 [00:01<00:00,  2.94it/s]
                 all         128         929       0.681       0.607       0.663       0.434

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
       2/2     7.75G   0.04461   0.06896   0.01866    0.1322       191       640: 100% 8/8 [00:02<00:00,  3.94it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100% 4/4 [00:03<00:00,  1.22it/s]
                 all         128         929       0.642       0.632       0.662       0.432
Optimizer stripped from runs/train/exp/weights/last.pt, 14.8MB
3 epochs completed in 0.007 hours.
123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172
```

### 4. 可视化 <a href="#t6" id="t6"></a>

### Weights \&Logging Biases 新 <a href="#t7" id="t7"></a>

[Weights & Biases](https://www.wandb.com/) (W\&B)现在与[YOLOv5](https://so.csdn.net/so/search?q=YOLOv5\&spm=1001.2101.3001.7020)用于实时可视化和训练运行的云记录。这样可以更好地进行运行比较和自省，并提高团队的可见性和协作性。启用W\&B `pip install wandb`,然后正常训练（首次使用时将指导您进行设置）。

在训练期间，您将在以下位置看到实时更新[https://wandb.ai/home](https://wandb.ai/home),您可以创建和共享详细信息[报告书](https://wandb.ai/glenn-jocher/yolov5\_tutorial/reports/YOLOv5-COCO128-Tutorial-Results--VmlldzozMDI5OTY)您的结果。有关更多信息，请参见[YOLOv5 Weights & Biases Tutorial](https://github.com/ultralytics/yolov5/issues/1289).

### 本地记录 <a href="#t8" id="t8"></a>

默认情况下，所有结果都记录在`runs/train`，并为每个新培训创建一个新的实验目录，如下所示：`runs/train/exp2`, `runs/train/exp3`，等等。查看火车并测试jpg，以查看镶嵌图案，标签，预测和增强效果。注意**镶嵌数据加载器**用于训练（如下所示），这是Ultralytics开发的新概念，首次在[YOLOv4](https://arxiv.org/abs/2004.10934).

```
Image(filename='runs/train/exp/train_batch0.jpg', width=800)  # train batch 0 mosaics and labels
Image(filename='runs/train/exp/test_batch0_labels.jpg', width=800)  # test batch 0 labels
Image(filename='runs/train/exp/test_batch0_pred.jpg', width=800)  # test batch 0 predictions
123
```

>

`train_batch0.jpg` shows train batch 0 mosaics and labels

>

`test_batch0_labels.jpg` shows test batch 0 labels

>

`test_batch0_pred.jpg` shows test batch 0 _predictions_

培训损失和绩效指标也被记录到[Tensorboard](https://www.tensorflow.org/tensorboard)以及一个自定义的`results.txt`日志文件，该文件以`results.png`形式在训练完成后绘制（如下）。在这里，我们展示了受过YOLOv5s在COCO128训练300 epochs,从头开始（蓝色），从预先训练`--weights yolov5s.pt` (橙色)。

```
from utils.plots import plot_results 
plot_results(save_dir='runs/train/exp')  # plot all results*.txt as results.png
Image(filename='runs/train/exp/results.png', width=800)
123
```

### 环境环境 <a href="#t9" id="t9"></a>

YOLOv5可以在以下任何最新的经过验证的环境中运行（所有依赖项包括[CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/)和[PyTorch](https://pytorch.org/)预装):

### 状态 <a href="#t10" id="t10"></a>

\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-a4Q0r8tE-1614831161591)(https://[github](https://so.csdn.net/so/search?q=github\&spm=1001.2101.3001.7020).com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg#pic\_center)]

如果此徽章为绿色，则全部[YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions)持续集成（CI）测试目前正在通过。\
CI测试验证YOLOv5培训的正确操作([train.py](https://github.com/ultralytics/yolov5/blob/master/train.py))，测试([test.py](https://github.com/ultralytics/yolov5/blob/master/test.py))，推论([detect.py](https://github.com/ultralytics/yolov5/blob/master/detect.py))和结果([export.py](https://github.com/ultralytics/yolov5/blob/master/models/export.py))在MacOS，Windows和Ubuntu上每隔24小时一次提交一次。

### 附录 <a href="#t11" id="t11"></a>

下面的可选附加功能。单元测试可验证回购功能，并且应在提交的任何PRs上运行。

```
# Re-clone repo
%cd ..
%rm -rf yolov5 && git clone https://github.com/ultralytics/yolov5
%cd yolov5
1234
```

```
# Reproduce
for x in 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x':
  !python test.py --weights {x}.pt --data coco.yaml --img 640 --conf 0.25 --iou 0.45  # speed
  !python test.py --weights {x}.pt --data coco.yaml --img 640 --conf 0.001 --iou 0.65  # mAP
1234
```

```
# Unit tests
%%shell
export PYTHONPATH="$PWD"  # to run *.py. files in subdirectories

rm -rf runs  # remove runs/
for m in yolov5s; do  # models
  python train.py --weights $m.pt --epochs 3 --img 320 --device 0  # train pretrained
  python train.py --weights '' --cfg $m.yaml --epochs 3 --img 320 --device 0  # train scratch
  for d in 0 cpu; do  # devices
    python detect.py --weights $m.pt --device $d  # detect official
    python detect.py --weights runs/train/exp/weights/best.pt --device $d  # detect custom
    python test.py --weights $m.pt --device $d # test official
    python test.py --weights runs/train/exp/weights/best.pt --device $d # test custom
  done
  python hubconf.py  # hub
  python models/yolo.py --cfg $m.yaml  # inspect
  python models/export.py --weights $m.pt --img 640 --batch 1  # export
done
123456789101112131415161718
```

```
# Profile
from utils.torch_utils import profile 

m1 = lambda x: x * torch.sigmoid(x)
m2 = torch.nn.SiLU()
profile(x=torch.randn(16, 3, 640, 640), ops=[m1, m2], n=100)
123456
```

```
# Evolve
!python train.py --img 640 --batch 64 --epochs 100 --data coco128.yaml --weights yolov5s.pt --cache --noautoanchor --evolve
!d=runs/train/evolve && cp evolve.* $d && zip -r evolve.zip $d && gsutil mv evolve.zip gs://bucket  # upload results (optional)
123
```

```
# VOC
for b, m in zip([64, 48, 32, 16], ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']):  # zip(batch_size, model)
  !python train.py --batch {b} --weights {m}.pt --data voc.yaml --epochs 50 --cache --img 512 --nosave --hyp hyp.finetune.yaml --project VOC --name {m}
123
```
