import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import dataloader
import image_util as kmg
import copy
import torch
import torchvision.models as models
import time
import copy
from collections import defaultdict
import pandas as pd
import numpy as np
import image_util as kmg
import dataloader
# Python's native libraries
import time
import os
import copy
from collections import defaultdict

# deep learning/vision libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import cv2 as cv  # OpenCV

# numeric and plotting libraries
import numpy as np
import matplotlib.pyplot as plt
from spatial_transforms import *
from utils import *
from torch.autograd import Variable
from torch.optim import lr_scheduler

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=700):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# def generate_model(model_depth, **kwargs):
#     assert model_depth in [10, 18, 34, 50, 101, 152, 200]

#     if model_depth == 10:
#         model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
#     elif model_depth == 18:
#         model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
#     elif model_depth == 34:
#         model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
#     elif model_depth == 50:
#         model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
#     elif model_depth == 101:
#         model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
#     elif model_depth == 152:
#         model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
#     elif model_depth == 200:
#         model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

#     parameters = get_fine_tuning_parameters(model, 'fc')
#     return model, parameters

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]

def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model
        if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
        else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)

    return model

def train(model,params):

    # transfer

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
    crop_method = crop_method = MultiScaleCornerCrop(
                5, 112, crop_positions=['c'])

    spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(1), norm_method
        ])
    temporal_transform = LoopPadding(10)
    target_transform = ClassLabel()

    train_data = Jester(
            r"D:/20bn-jester-v2",
            "jester.json",
            'train',
            10,
             spatial_transform= Compose([
    #Scale(opt.sample_size),
    Scale(112),
    CenterCrop(112),
    ToTensor(1), Normalize([0, 0, 0], [1, 1, 1]) ]),
            temporal_transform= LoopPadding(16),
            target_transform=ClassLabel(),
            sample_duration=16)
    train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=True)
    
    train_logger = Logger(
            os.path.join('result', 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(
            os.path.join('result', 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    spatial_transform = Compose([
            Scale(112),
            CenterCrop(112),
            ToTensor(1), norm_method
        ])
    temporal_transform = LoopPadding(10)
    target_transform = ClassLabel()
    validation_data = Jester(
            r"D:/20bn-jester-v2",
            "jester.json",
            'validation',
            10,
            spatial_transform= Compose([
    #Scale(opt.sample_size),
    Scale(112),
    CenterCrop(112),
    ToTensor(1), Normalize([0, 0, 0], [1, 1, 1]) ]),
            temporal_transform= TemporalCenterCrop(16,1),
            target_transform=ClassLabel(),
            sample_duration=16)
    val_loader =  torch.utils.data.DataLoader(
            train_data,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=True)
    val_logger = Logger(
        os.path.join('result', 'val.log'), ['epoch', 'loss', 'acc'])

    optimizer = optim.SGD(
            params,
            lr=0.1,
            momentum=0.9,
            dampening=0.9,
            weight_decay=1e-3,
            nesterov=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=10)


    # val_logger = Logger(
    #         os.path.join('results', 'val.log'), ['epoch', 'loss', 'acc'])

    try:
        print('loading checkpoint {}'.format(''))
        checkpoint = torch.load('')
        begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        begin_epoch = 1
    
    print('run')
    for i in range(begin_epoch,200 + 1):
        train_epoch(i, train_loader, model, criterion, optimizer, 
                        train_logger, train_batch_logger)
        # validation_loss = val_epoch(i, val_loader, model, criterion, 
        #                                 val_logger)

        #scheduler.step(validation_loss[0])
        scheduler.step(70)

    # test
    spatial_transform = Compose([
        Scale(int(112 / 1)),
        CornerCrop(112, 'c'),
        ToTensor(1), norm_method
    ])
    temporal_transform = LoopPadding(10)
    target_transform=ClassLabel()

    test_data = Jester(
            r"D:/20bn-jester-v2",
            "jester.json",
            'test',
            10,
             spatial_transform= Compose([
    #Scale(opt.sample_size),
    Scale(112),
    CenterCrop(112),
    ToTensor(1), Normalize([0, 0, 0], [1, 1, 1]) ]),
            temporal_transform= TemporalCenterCrop(16,1),
            target_transform=ClassLabel(),
            sample_duration=16)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    #test.test(test_loader, model,  test_data.class_names)

def train_epoch(epoch, data_loader, model, criterion, optimizer, 
                epoch_logger, batch_logger):
    print('train at epoch '+str(epoch))

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):

        our_gestures = [15,16,17,18,19]
        nothing = [0,2]
        transf_targets = []
        for t in targets:
            if t in our_gestures:
                val = 0
            elif t in nothing:
                val = 1
            else:
                val = 2
            transf_targets.append(val)
        targets = torch.LongTensor(transf_targets)

        data_time.update(time.time() - end_time)
        
        inputs = Variable(inputs)
        targets = Variable(targets)
        targets = targets.to(device)
        inputs = inputs.to(device)
       
        outputs = model(inputs)
        outputs = outputs.to(device)

        loss = criterion(outputs, targets)
        loss = loss.to(device)
        acc = calculate_accuracy(outputs.data, targets.data)

        print(acc)
        print(accuracies)

        #losses.update(loss.data[0], inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc[0].item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % 1 == 0:
        save_file_path = os.path.join('result',
                                      '01i00_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

def val_epoch(epoch, data_loader, model, criterion, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        our_gestures = [15,16,17,18,19]
        nothing = [0,2]
        transf_targets = []
        for t in targets:
            if t in our_gestures:
                val = 0
            elif t in nothing:
                val = 1
            else:
                val = 2
            transf_targets.append(val)
        targets = torch.LongTensor(transf_targets)

        targets = targets.to(device)
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.to(device)
        loss = criterion(outputs, targets)
        loss = outputs.to(loss)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        losses.update(loss.data, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 10 ==0:
            print('Epoch ' + str(epoch))
            print('Batch ' + str(i+1))
            print('Len ' + str(len(data_loader)))
            print('Top1 prec ' + str(top1.avg.item()))
            print('Top5 prec' + str(top5.avg.item()))

        #   print('Epoch: [{0}][{1}/{2}]\t'
        #       'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
        #       'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #       'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
        #       'Prec@5 {top5val:.5f} ({top5avg:.5f})'.format(
        #           epoch,
        #           i + 1,
        #           len(data_loader),
        #           batch_time=batch_time,
        #           data_time=data_time,
        #           loss=losses,
        #           top1=top1,
        #           top5val=top5.val[0].item(),
        #           top5avg=top5.avg[0].item()
        #           ))

    # logger.log({'epoch': epoch,
    #             'loss': losses.avg.item(),
    #             'prec1': top1.avg.item(),
    #             'prec5': top5.avg.item()})

    # return losses.avg.item(), top1.avg.item()

    val_logger = Logger(
        os.path.join('result', 'val.log'), ['epoch', 'loss', 'acc'])

    val_logger.log({'epoch': epoch,
                'loss': losses.avg,
                'acc': top1.avg
                })

    return losses.avg, top1.avg


def test(model,params):

    path_vid =r"D:/20bn-jester-v2"
    BASE_PATH = "D:"

    test_data = Jester(
            path_vid,
            "jester.json",
            'test',
            10,
            spatial_transform= Compose([
    #Scale(opt.sample_size),
    Scale(112),
    CenterCrop(112),
    ToTensor(1), Normalize([0, 0, 0], [1, 1, 1]) ]),
            temporal_transform= TemporalCenterCrop(16,1),
            target_transform=ClassLabel(),
            sample_duration=16)

    test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    recorder = []

    model.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    y_true = []
    y_pred = []
    end_time = time.time()
    print(test_loader)
    for i, (inputs, targets) in enumerate(test_loader):
        # if (i==200):
        #     break
        print(str(i)+'/'+str(len(test_loader)))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        start = time.time()
        our_gestures = [15,16,17,18,19]
        nothing = [0,2]
        transf_targets = []
        for t in targets:
            if t in our_gestures:
                val = 0
            elif t in nothing:
                val = 1
            else:
                val = 2
            transf_targets.append(val)
        targets = torch.LongTensor(transf_targets)

        #inputs = Variable(torch.squeeze(inputs), volatile=True)
        model.eval()
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
            start = time.time()
            inputs = inputs.to(device)
            targets = targets.to(device)
            model.to(device)
            outputs = model(inputs)
            print('Time = ' + str(time.time()-start))
            #outputs = F.softmax(outputs)
            #print('Time = ' + str(time.time()-start))
            recorder.append(outputs.data.cpu().numpy().copy())
            ##print(targets)
            #print(outputs.argmax(1))
        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        # print('[{0}/{1}]\t'
        #         'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
        #         'prec@1 {top1.avg:.5f} prec@5 {top5.avg:.5f}\t'
        #         'precision {precision.val:.5f} ({precision.avg:.5f})\t'
        #         'recall {recall.val:.5f} ({recall.avg:.5f})'.format(
        #             i + 1,
        #             len(test_loader),
        #             batch_time=batch_time,
        #             top1 =top1,
        #             top5=top5,
        #             precision = precisions,
        #             recall = recalls)
       
        
        prec1, prec5 = calculate_accuracy(outputs, targets, topk=(1,2))

        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

    print(len(test_loader))
    print(top1.avg)
    print(prec1.item())
        

# model_shell, parameters = generate_model(18, n_classes=700)
# model = load_pretrained_model(model_shell, r"D:\TetrisProject\r3d18_K_200ep.pth", 'resnet',27)
model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(),  n_classes=3)
pretrain = torch.load(r"result\21i04_1.pth")
model.load_state_dict(pretrain['state_dict'])
# model.fc = nn.Linear(model.fc.in_features,
#                                3)
model.fc = model.fc.cuda()

parameters = get_fine_tuning_parameters(model, 4)
#parameters = get_fine_tuning_parameters(model,0)
#model, parameters = generate_model(18)
#train(model,parameters)
test(model, parameters)