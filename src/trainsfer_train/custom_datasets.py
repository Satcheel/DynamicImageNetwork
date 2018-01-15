import torch.utils.data as data
import numpy
from PIL import Image
import os
import os.path
import trainsfer_train.config as config
import torch


# 默认的图片读取方式
def default_loader(path):
    return Image.open(path).convert('RGB')


# 读取列表文件，文件的每一行应该是path label
# 仅支持单标签
def default_flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))
    return imlist


# 支持同一视频多张图片读取
class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, repeat=1):
        self.root = os.path.expanduser(root)
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.repeat = repeat

    # 输出tensor的维度
    # 如果只有一张图片 (n_channels,height,width)
    # 如果有多张图片 (n_pics,n_channels,height,width)
    def __getitem__(self, index):
        impath, target = self.imlist[index]
        imgs = []
        if self.repeat == 1:
            specific_impath = impath + "_00" + ".jpg"
            imgs = self.loader(os.path.join(self.root, specific_impath))
            if self.transform is not None:
                imgs = self.transform(imgs)
        else:
            for i in range(self.repeat):
                specific_impath = impath + "_{:02d}".format(i) + ".jpg"
                img = self.loader(os.path.join(self.root, specific_impath))
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)
            imgs = torch.stack(imgs)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs, target

    def __len__(self):
        return len(self.imlist)


class CustomDataset(object):
    def __init__(self):
        self.specific_config = None

    def __call__(self, *args, **kwargs):
        if kwargs.get("train", True):
            return ImageFilelist(
                root=self.specific_config.Train.directory,
                flist=self.specific_config.Train.label,
                transform=kwargs.get("transform", None),
                repeat=kwargs.get("repeat", 1))
        else:
            return ImageFilelist(
                root=self.specific_config.Validation.directory,
                flist=self.specific_config.Validation.label,
                transform=kwargs.get("transform", None),
                repeat=kwargs.get("repeat", 1))


class MaxImageDataset(CustomDataset):
    def __init__(self):
        super(MaxImageDataset, self).__init__()
        self.specific_config = config.Dataset.MaxImage


class MeanImageDataset(CustomDataset):
    def __init__(self):
        super(MeanImageDataset, self).__init__()
        self.specific_config = config.Dataset.MeanImage


class StaticImageDataset(CustomDataset):
    def __init__(self):
        super(StaticImageDataset, self).__init__()
        self.specific_config = config.Dataset.StaticImage


class MDIDataset(CustomDataset):
    def __init__(self):
        super(MDIDataset, self).__init__()
        self.specific_config = config.Dataset.MDIImage


class SDIDataset(CustomDataset):
    def __init__(self):
        super(SDIDataset, self).__init__()
        self.specific_config = config.Dataset.SDIImage
