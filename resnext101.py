import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.nn.functional as F
import dataloader
import image_util as kmg
import copy
import torchvision.models as models
import time
from collections import defaultdict
import pandas as pd
import numpy as np
import image_util as kmg
import dataloader
# Python's native libraries
import os
from collections import defaultdict
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import cv2 as cv  # OpenCV
# numeric and plotting libraries
import matplotlib.pyplot as plt
import jester
from PIL import Image
from spatial_transforms import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from utils import *


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
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


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        #self.conv1 = nn.Conv3d(
        #    3,
        #    64,
        #    kernel_size=(3,7,7),
        #    stride=(1, 2, 2),
        #    padding=(1, 3, 3),
        #    bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        #last_duration = 1
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def resnext50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model

# def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
#     if pretrain_path:
#         #print('loading pretrained model {}'.format(pretrain_path))
#         pretrain = torch.load(pretrain_path, map_location='cpu')

#         model.load_state_dict(pretrain['state_dict'])
#         tmp_model = model
#         if model_name == 'densenet':
#             tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
#                                              n_finetune_classes)
#         else:
#             tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
#                                      n_finetune_classes)

#     return model

def load():
    path =  r"jester_resnext_101_RGB_16_best.pth"

 
#        assert opt.arch == checkpoint['arch']

    model =  ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3],
                num_classes=27,
                shortcut_type='B',
                cardinality=32,
                sample_size=112,
                sample_duration=16)
    
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()

    dicti = pretrained_dict['state_dict']

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in dicti.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params

    # 2. overwrite entries in the existing state dict
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    return model

def test(model):

    path_vid =r"D:/20bn-jester-v2"
    BASE_PATH = "D:"
    #path_model = os.path.join(data_root, data_model, model_name)
    path_labels = BASE_PATH + '/jester-v1-labels.csv'
    path_train =  BASE_PATH + '/jester-v1-train.csv'
    path_test =  BASE_PATH + '/jester-v1-test.csv'
    path_val = BASE_PATH + '/jester-v1-validation.csv'

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
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    recorder = []

    model.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    precisions = AverageMeter() #
    recalls = AverageMeter()
    
    y_true = []
    y_pred = []
    end_time = time.time()
    print(test_loader)
    for i, (inputs, targets) in enumerate(test_loader):
        if i == 5:
            break
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        targets = targets.to(device)
        #inputs = Variable(torch.squeeze(inputs), volatile=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
            start = time.time()
            outputs = model(inputs)
            #outputs = F.softmax(outputs)
            print('Time = ' + str(time.time()-start))
            recorder.append(outputs.data.cpu().numpy().copy())
            print(targets)
            print(outputs.argmax(1))
        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())
    print(y_true)
    print(y_pred)
    
    prec1, prec5 = calculate_accuracy(outputs, targets, topk=(1,5))
    precision = calculate_precision(outputs, targets) #
    recall = calculate_recall(outputs,targets)


    top1.update(prec1, inputs.size(0))
    top5.update(prec5, inputs.size(0))
    precisions.update(precision, inputs.size(0))
    recalls.update(recall,inputs.size(0))

    batch_time.update(time.time() - end_time)
    end_time = time.time()
    print('[{0}/{1}]\t'
            'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
            'prec@1 {top1.avg:.5f} prec@5 {top5.avg:.5f}\t'
            'precision {precision.val:.5f} ({precision.avg:.5f})\t'
            'recall {recall.val:.5f} ({recall.avg:.5f})'.format(
                i + 1,
                len(test_loader),
                batch_time=batch_time,
                top1 =top1,
                top5=top5,
                precision = precisions,
                recall = recalls))


def calculate_accuracy(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    ret = []
    for k in topk:
        correct_k = correct[:k].float().sum().item()
        ret.append(correct_k / batch_size)

    return ret
def calculate_precision(outputs, targets):

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return  precision_score(targets.view(-1), pred.view(-1), average = 'macro')


def calculate_recall(outputs, targets):

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return  recall_score(targets.view(-1), pred.view(-1), average = 'macro')

model = load()
test(model)