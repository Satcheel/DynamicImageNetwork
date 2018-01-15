# 本文件用来计算数据集的均值和标准差

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print(f'==> Computinlsg mean and std for dataset {dataset.root}')
    for index,(inputs, targets) in enumerate(dataloader):

        for i in range(3):
            print(f"calculating pic {index} mean={inputs[:,i,:,:].mean()}")
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])

    datasets = [
        ImageFolder(root="/home/chenyaofo/datasets/UCF_extract/MaxImage",transform=tf),
        ImageFolder(root="/home/chenyaofo/datasets/UCF_extract/MeanImage", transform=tf),
        ImageFolder(root="/home/chenyaofo/datasets/UCF_extract/StaticImage", transform=tf),
        ImageFolder(root="/home/chenyaofo/datasets/UCF_extract/MDIImage", transform=tf),
        ImageFolder(root="/home/chenyaofo/datasets/UCF_extract/SDIImage", transform=tf),
    ]

    for dataset in datasets:
        mean,std = get_mean_and_std(dataset)
        filename=  os.path.join(dataset.root,"mean_std.txt")
        with open(filename,"w") as f:
            f.write(f"mean={mean.tolist()}\n")
            f.write(f"std={std.tolist()}\n")