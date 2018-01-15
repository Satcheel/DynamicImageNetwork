## 本文件是训练入口

import torch.nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from trainsfer_train.custom_datasets import *
import math
import argparse
import pickle
import os
from torchnet.meter import mAPMeter
from trainsfer_train.util import *
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name", type=str)
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument("--pretrain", dest="pretrain", action="store_true")
    flag_parser.add_argument("--no-pretrain", dest="pretrain", action="store_false")
    args = parser.parse_args()
    print(f"Training start with dataset {args.name}.")

    datasets_dict = dict(
        Max=MaxImageDataset(),
        Mean=MeanImageDataset(),
        Static=StaticImageDataset(),
        SDI=SDIDataset(),
        MDI=MDIDataset(),
    )
    recoder = defaultdict(list)
    recoder["name"] = args.name
    dataset = datasets_dict.get(args.name, None)
    if args.pretrain:
        normalization = [transforms.Normalize(mean=dataset.specific_config.mean, std=dataset.specific_config.std)]
    else:
        # use the normalized parameters on http://pytorch.org/docs/0.3.0/torchvision/models.html
        normalization = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    tf = transforms.Compose([
                                transforms.Resize(size=(224, 224)),
                                transforms.ToTensor(),
                            ] + normalization)
    train_loader = DataLoader(
        dataset=dataset(train=True, transform=tf),
        batch_size=config.Strategy.Train.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        dataset=dataset(train=False, transform=tf),
        batch_size=config.Strategy.Validation.batch_size,
        shuffle=True,
        num_workers=2,
    )
    if args.pretrain:
        print("use pretrain resnet18")
    else:
        print("use unpretrain resnet18")
    net = models.resnet18(pretrained=args.pretrain)
    fc_in_features = net.fc.in_features
    net.fc = torch.nn.Linear(in_features=fc_in_features, out_features=101, )

    cudable = torch.cuda.is_available()
    if args.pretrain:
        ignored_params = list(map(id, net.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             net.parameters())
        optimizer = torch.optim.SGD([
            dict(params=base_params, lr=config.Strategy.Train.learning_rate / 10),
            dict(params=net.fc.parameters())
        ], lr=config.Strategy.Train.learning_rate, momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=config.Strategy.Train.learning_rate, momentum=0.9,
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
        return loss_scaler, accuracy #meter.value()


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
        return loss_scaler, accuracy #meter.value()


    max_epoch = config.Strategy.Train.max_epoches
    for epoch in range(max_epoch):
        loss, acc= train(epoch)
        recoder["train_loss"].append(loss)
        recoder["train_accuracy"].append(acc)

        loss, acc= validate(epoch)
        recoder["val_loss"].append(loss)
        recoder["val_accuracy"].append(acc)


    os.makedirs("result", exist_ok=True)
    filename = f"{args.name}_{'pretrain' if args.pretrain else 'unpretrain'}"
    with open(f"result/{filename}.static", "wb") as f:
        pickle.dump(recoder, f)
