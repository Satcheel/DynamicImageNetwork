# 本工程为对论文*Dynamic Image Network for Action Recognition*的复现。

### 分为以下三个模块：
 - video compression
 这个模块主要是负责提取视频的信息并将其压缩为图片，图片种类包括论文中的Max Image, Mean Image, Static Image, SDI Image(对整个视频压缩得到单张图片)和MDI Image(对视频分段进行压缩得到多张图片)。
 - create dataset lists
 官方给的训练集和验证集列表存在一定的问题：1. 官方给的标注是从1到101，pytorch要求标注从0开始； 2. 官方列表中文件的后缀名是avi，在本工程中是jpg。所以在这个模块中对数据集列表进行简单的预处理。
 - transfer train
 这个模块使用torchvision中的resnet18进行迁移训练。因为MDI的训练与其他图片训练存在较大区别，因此有单独的一个文件MDI_train.py用来训练。
 
### 注意
本工程所有运行代码请加入环境变量才能运行，PYTHONPATH=path/to/DynamicImageNetwork/src。否则包间关系导入会出错。