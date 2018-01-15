# 本文件是训练MDI的入口

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from trainsfer_train.custom_datasets import *
import math
import pickle
import os
import argparse
from collections import defaultdict
from trainsfer_train.util import *


class EarlyFusionResNet(nn.Module):
    def __init__(self, pretrain, repeat):
        super(EarlyFusionResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrain)
        self.resnet.conv1 = nn.Conv2d(3 * repeat, 64, kernel_size=7, stride=2, padding=3, bias=False)
        fc_in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(in_features=fc_in_features, out_features=101, )

        self.new_layers = nn.Sequential(self.resnet.conv1, self.resnet.fc)
        self.new_layer_param_ids = list(map(id, self.new_layers.parameters()))
        self.old_layer_params = filter(lambda p: id(p) not in self.new_layer_param_ids, self.resnet.parameters())

    def forward(self, x):
        n_samples, n_repeat, n_channels, height, width = x.size()
        x = x.view(n_samples, n_repeat * n_channels, height, width)
        return self.resnet(x)


class LateFusionResNet(nn.Module):
    def __init__(self, pretrain, repeat):
        super(LateFusionResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrain)
        fc_in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(in_features=fc_in_features, out_features=101, )
        self.new_layers = self.resnet.fc
        self.new_layer_param_ids = list(map(id, self.new_layers.parameters()))
        self.old_layer_params = filter(lambda p: id(p) not in self.new_layer_param_ids, self.resnet.parameters())

    def forward(self, x):
        n_samples, n_repeat, n_channels, height, width = x.size()
        x = x.view(n_samples * n_repeat, n_channels, height, width)
        x = self.resnet(x)
        x = x.view(n_samples, n_repeat, -1)
        x = x.mean(dim=1, keepdim=False)
        return x


# PYTHONPATH=/home/chenyaofo/workspace/DynamicImageNetwork/src CUDA_VISIBLE_DEVICES=0 python MDI_train.py --no-early --no-pretrain

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 命令行参数，决定是否使用预训练的网络
    flag_parser = parser.add_mutually_exclusive_group(required=True)
    flag_parser.add_argument("--pretrain", dest="pretrain", action="store_true")
    flag_parser.add_argument("--no-pretrain", dest="pretrain", action="store_false")
    # 命令行参数，决定使用EarlyFusionResNet还是LateFusionResNet
    flag_parser = parser.add_mutually_exclusive_group(required=True)
    flag_parser.add_argument("--early", dest="early", action="store_true")
    flag_parser.add_argument("--no-early", dest="early", action="store_false")
    args = parser.parse_args()
    name = "MDI"
    print(f"Use dataset {name}.")
    datasets_dict = dict(
        MDI=MDIDataset(),
    )
    recoder = defaultdict(list)
    recoder["name"] = name
    tf = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5098264813423157, 0.5102994441986084, 0.509983479976654],
                             std=[0.08257845789194107, 0.07896881550550461, 0.07860155403614044])
    ])
    dataset = datasets_dict.get(name, None)
    train_loader = DataLoader(
        dataset=dataset(train=True, transform=tf, repeat=3),
        batch_size=config.Strategy.Train.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        dataset=dataset(train=False, transform=tf, repeat=3),
        batch_size=config.Strategy.Validation.batch_size,
        shuffle=True,
        num_workers=2,
    )
    if args.pretrain:
        print("Use pretrain resnet18.")
    else:
        print("Use unpretrain resnet18.")
    if args.early:
        net = EarlyFusionResNet(pretrain=args.pretrain, repeat=3)
        print("Network init with EarlyFusionResNet.")
    else:
        net = LateFusionResNet(pretrain=args.pretrain, repeat=3)
        print("Network init with LateFusionResNet.")

    cudable = torch.cuda.is_available()
    if args.pretrain:
        optimizer = torch.optim.SGD([
            dict(params=net.old_layer_params, lr=config.Strategy.Train.learning_rate / 10),
            dict(params=net.new_layers.parameters())
        ], lr=config.Strategy.Train.learning_rate, momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = torch.optim.SGD(net.resnet.parameters(), lr=config.Strategy.Train.learning_rate, momentum=0.9,
                                    weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    if cudable:
        net = net.cuda()


    def train(epoch):
        print("TRAIN epoch={:03d}".format(epoch))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        loss_scaler = 0.0
        accuracy = 0.0
        # meter = mAPMeter()
        for batch_index, (data, targets) in enumerate(train_loader, start=0):
            if cudable:
                data, targets = data.cuda(), targets.cuda()
            data, targets = Variable(data), Variable(targets)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # meter.add(outputs.data, targets.data)

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            loss_scaler = train_loss / (batch_index + 1)
            accuracy = correct / total
            progress_bar(batch_index, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (loss_scaler, 100. * accuracy, correct, total))

        scheduler.step()
        return loss_scaler, accuracy  # meter.value()


    def validate(epoch):
        print("TEST  epoch={:03d}".format(epoch))
        net.eval()
        val_loss = 0
        correct = 0
        total = 0
        loss_scaler = 0.0
        accuracy = 0.0
        # meter = mAPMeter()
        for batch_index, (data, targets) in enumerate(val_loader, start=0):
            if cudable:
                data, targets = data.cuda(), targets.cuda()
            data, targets = Variable(data), Variable(targets)
            outputs = net(data)
            loss = criterion(outputs, targets)

            # meter.add(outputs.data, targets.data)

            val_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            loss_scaler = val_loss / (batch_index + 1)
            accuracy = correct / total

            progress_bar(batch_index, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (loss_scaler, 100. * accuracy, correct, total))
        return loss_scaler, accuracy  # meter.value()


    max_epoch = config.Strategy.Train.max_epoches
    for epoch in range(max_epoch):
        loss, acc = train(epoch)
        recoder["train_loss"].append(loss)
        recoder["train_accuracy"].append(acc)

        loss, acc = validate(epoch)
        recoder["val_loss"].append(loss)
        recoder["val_accuracy"].append(acc)

    # 保存实验结果
    recoder["name"] = f"{name}-{'E' if args.early else 'L'}"
    os.makedirs("result", exist_ok=True)
    filename = f"{name}-{'E' if args.early else 'L'}_{'pretrain' if args.pretrain else 'unpretrain'}"
    with open(f"result/{filename}.static", "wb") as f:
        pickle.dump(recoder, f)
