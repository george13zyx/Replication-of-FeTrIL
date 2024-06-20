#!/usr/bin/env python
# coding=utf-8
import time
import shutil
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.optim.optimizer import Optimizer
import torch.utils.model_zoo as model_zoo
import torch.utils.data as data
from collections import defaultdict
from datetime import timedelta
from configparser import ConfigParser
import numpy as np
from PIL import Image
import imghdr
import pandas as pd
import os
import os.path
import random
import multiprocessing as mp
import sys
import re
import math
import modified_linear


sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'utilsCIL'))
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'FeTrIL'))

class AverageMeter(object):
    # 初始化函数，创建一个新的平均计量器
    def __init__(self):
        # 调用reset函数初始化各项参数
        self.reset()

    # 重置计量器参数
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # 更新计量器参数
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

# 检查文件名的扩展名是否在指定扩展名列表中
def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

# 在指定路径下处理图像文件列表
def onscratch(images_list_file):
    dataset = images_list_file.split('/')[-2]
    data_type = 'train' if 'train.lst' in images_list_file else 'test'
    df = pd.read_csv(images_list_file, sep=' ', names=['paths', 'class'])
    root_folder = df['paths'].head(1)[0]
    df = df.tail(df.shape[0] - 1)
    df.drop_duplicates()
    df = df.sort_values('class')
    train_list = df['paths'].tolist()
    for elt in train_list:
        parent_dir = os.path.join('/sscratch/dataset/', dataset, os.path.dirname(elt))
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

    # 复制图像文件到指定目录
    def copy_image(image):
        shutil.copy(image, os.path.join('/sscratch/dataset/', dataset))

    # 使用多进程池并行处理图像复制
    with mp.Pool() as pool:
        pool.map(copy_image, train_list)
    print(images_list_file, 'copied to /sscratch/dataset/', dataset)

class ImagesListFileFolder(data.Dataset):

    def __init__(self, images_list_file, transform=None, target_transform=None, return_path=False, range_classes=None,
                 random_seed=-1, old_load=False, open_images=False, nb_classes=None, onscratch=False, verbose=True):
        # 初始化数据集对象
        self.return_path = return_path
        samples = []
        df = pd.read_csv(images_list_file, sep=' ', names=['paths', 'class'])
        if old_load:
            root_folder = ''
        else:
            root_folder = df['paths'].head(1)[0]
            if onscratch:
                dataset = images_list_file.split('/')[-2]
                root_folder = os.path.join('/sscratch/dataset/', dataset)
            df = df.tail(df.shape[0] - 1)
        df.drop_duplicates()
        df['class'] = df['class'].astype(int)
        df = df.sort_values('class')
        if nb_classes:
            order_list = [i for i in range(nb_classes)]
        else:
            order_list = [i for i in range(1 + max(list(set(df['class'].values.tolist()))))]
        if verbose:
            print('*' * (len(images_list_file) + 76 + len(str(random_seed))))
            print('Class order of', images_list_file, 'before shuffle with seed', random_seed, ': [', *order_list[:5],
                  '...', *order_list[-5:], ']')
        if random_seed != -1:
            np.random.seed(random_seed)
            random.shuffle(order_list)
            order_list = np.random.permutation(len(order_list)).tolist()
        if verbose:
            print('Class order of', images_list_file, 'after  shuffle with seed', random_seed, ': [', *order_list[:5],
                  '...', *order_list[-5:], ']')
            print('*' * (len(images_list_file) + 76 + len(str(random_seed))))
        order_list_reverse = [order_list.index(i) for i in list(set(df['class'].values.tolist()))]
        if range_classes:
            index_to_take = [order_list[i] for i in range_classes]
            samples = [(os.path.join(root_folder, elt[0]), order_list_reverse[elt[1]]) for elt in
                       list(map(tuple, df.loc[df['class'].isin(index_to_take)].values.tolist()))]
            samples.sort(key=lambda x: x[1])
            if verbose:
                print('We pick the classes from', min(range_classes), 'to', max(range_classes), 'and we have',
                      len(samples), 'samples')
                print('We have', len(list(set(df['class'].values.tolist()))), 'classes in total')
        else:
            samples = [(os.path.join(root_folder, elt[0]), order_list_reverse[elt[1]]) for elt in
                       list(map(tuple, df.values.tolist()))]
            samples.sort(key=lambda x: x[1])
        if not samples:
            raise (RuntimeError("No image found"))

        self.images_list_file = images_list_file
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = [s[0] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # 获取指定索引的样本和标签
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_path:
            return (sample, target), self.samples[index][0]
        return sample, target

    def __len__(self):
        # 返回数据集长度
        return len(self.samples)

    def __repr__(self):
        # 返回数据集描述
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    List Description: {}\n'.format(self.images_list_file)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class ImagesListFileFolderMixup(data.Dataset):

    def __init__(self, images_list_file, transform=None, target_transform=None, return_path=False, range_classes=None,
                 random_seed=-1, old_load=False, mixup_alpha=0.2, one_every=5):

        self.return_path = return_path
        self.mixup_alpha = mixup_alpha
        self.one_every = one_every
        samples = []
        df = pd.read_csv(images_list_file, sep=' ', names=['paths', 'class'])
        if old_load:
            root_folder = ''
        else:
            root_folder = df['paths'].head(1)[0]
            df = df.tail(df.shape[0] - 1)
        df.drop_duplicates()
        df['class'] = df['class'].astype(int)
        df = df.sort_values('class')
        order_list = [i for i in range(1 + max(list(set(df['class'].values.tolist()))))]
        print('*' * (len(images_list_file) + 76 + len(str(random_seed))))
        print('Class order of', images_list_file, 'before shuffle with seed', random_seed, ': [', *order_list[:5],
              '...', *order_list[-5:], ']')
        if random_seed != -1:
            np.random.seed(random_seed)
            random.shuffle(order_list)
            order_list = np.random.permutation(len(order_list)).tolist()
        print('Class order of', images_list_file, 'after  shuffle with seed', random_seed, ': [', *order_list[:5],
              '...', *order_list[-5:], ']')
        print('*' * (len(images_list_file) + 76 + len(str(random_seed))))
        order_list_reverse = [order_list.index(i) for i in list(set(df['class'].values.tolist()))]
        if range_classes:
            index_to_take = [order_list[i] for i in range_classes]
            samples = [(os.path.join(root_folder, elt[0]), order_list_reverse[elt[1]]) for elt in
                       list(map(tuple, df.loc[df['class'].isin(index_to_take)].values.tolist()))]
            samples.sort(key=lambda x: x[1])
            print('We pick the classes from', min(range_classes), 'to', max(range_classes), 'and we have', len(samples),
                  'samples')
        else:
            samples = [(os.path.join(root_folder, elt[0]), order_list_reverse[elt[1]]) for elt in
                       list(map(tuple, df.values.tolist()))]
            samples.sort(key=lambda x: x[1])
        if not samples:
            raise (RuntimeError("No image found"))

        self.images_list_file = images_list_file
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = [s[0] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
        self.total_classes = 1 + max(list(set(df['class'].values.tolist())))

    def __getitem__(self, index):
        path, target = self.samples[index]
        label = torch.zeros(self.total_classes)
        label[target] = 1.

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if index % self.one_every == 0:
            mixup_idx = random.choice([i for i in range(len(self.samples)) if i != index])
            mixup_label = torch.zeros(self.total_classes)
            label[self.targets[mixup_idx]] = 1.
            if self.transform is not None:
                mixup_image = self.transform(self.loader(self.samples[mixup_idx][0]))
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            sample = lam * sample + (1 - lam) * mixup_image
            label = lam * label + (1 - lam) * mixup_label

        if self.return_path:
            return (sample, label), self.samples[index][0]
        return sample, label

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    List Description: {}\n'.format(self.images_list_file)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class ImagesListFileFolderMixup(data.Dataset):

    def __init__(self, images_list_file, transform=None, target_transform=None, return_path=False, range_classes=None,
                 random_seed=-1, old_load=False, mixup_alpha=0.2, one_every=5):
        # 初始化函数：设置数据集属性，加载图像列表文件，并进行预处理
        self.return_path = return_path
        self.mixup_alpha = mixup_alpha
        self.one_every = one_every
        samples = []
        df = pd.read_csv(images_list_file, sep=' ', names=['paths', 'class'])
        if old_load:
            root_folder = ''
        else:
            root_folder = df['paths'].head(1)[0]
            df = df.tail(df.shape[0] - 1)
        df.drop_duplicates()
        df['class'] = df['class'].astype(int)
        df = df.sort_values('class')
        order_list = [i for i in range(1 + max(list(set(df['class'].values.tolist()))))]
        print('*' * (len(images_list_file) + 76 + len(str(random_seed))))
        print('Class order of', images_list_file, 'before shuffle with seed', random_seed, ': [', *order_list[:5],
              '...', *order_list[-5:], ']')
        if random_seed != -1:
            np.random.seed(random_seed)
            random.shuffle(order_list)
            order_list = np.random.permutation(len(order_list)).tolist()
        print('Class order of', images_list_file, 'after  shuffle with seed', random_seed, ': [', *order_list[:5],
              '...', *order_list[-5:], ']')
        print('*' * (len(images_list_file) + 76 + len(str(random_seed))))
        order_list_reverse = [order_list.index(i) for i in list(set(df['class'].values.tolist()))]
        if range_classes:
            index_to_take = [order_list[i] for i in range_classes]
            samples = [(os.path.join(root_folder, elt[0]), order_list_reverse[elt[1]]) for elt in
                       list(map(tuple, df.loc[df['class'].isin(index_to_take)].values.tolist()))]
            samples.sort(key=lambda x: x[1])
            print('We pick the classes from', min(range_classes), 'to', max(range_classes), 'and we have', len(samples),
                  'samples')
        else:
            samples = [(os.path.join(root_folder, elt[0]), order_list_reverse[elt[1]]) for elt in
                       list(map(tuple, df.values.tolist()))]
            samples.sort(key=lambda x: x[1])
        if not samples:
            raise (RuntimeError("No image found"))

        self.images_list_file = images_list_file
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = [s[0] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
        self.total_classes = 1 + max(list(set(df['class'].values.tolist())))

    def __getitem__(self, index):
        # 获取数据集中指定索引处的图像及其标签，并进行Mixup增强
        path, target = self.samples[index]
        label = torch.zeros(self.total_classes)
        label[target] = 1.

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if index % self.one_every == 0:
            mixup_idx = random.choice([i for i in range(len(self.samples)) if i != index])
            mixup_label = torch.zeros(self.total_classes)
            label[self.targets[mixup_idx]] = 1.
            if self.transform is not None:
                mixup_image = self.transform(self.loader(self.samples[mixup_idx][0]))
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            sample = lam * sample + (1 - lam) * mixup_image
            label = lam * label + (1 - lam) * mixup_label

        if self.return_path:
            return (sample, label), self.samples[index][0]
        return sample, label

    def __len__(self):
        # 返回数据集的大小（样本数）
        return len(self.samples)

    def __repr__(self):
        # 返回数据集的描述信息，包括数据集名称、数据集大小和数据转换信息
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    List Description: {}\n'.format(self.images_list_file)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

# 从文件路径加载图片并转换为RGB格式
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# 使用accimage加载图片，若失败则调用pil_loader函数
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

# 根据当前环境选择合适的图片加载函数：如果支持accimage则调用accimage_loader否则调用pil_loader
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# 自定义数据集类，从给定的图像列表文件中加载数据集，并根据提供的索引和返回项索引进行处理
class ImagesListFolderIndexRemind(data.Dataset):
    def __init__(self, images_list_file, indices, return_item_ix, transform=None, target_transform=None):
        # 初始化函数，读取图像列表文件并准备数据集
        samples = []
        df = pd.read_csv(images_list_file, sep=' ', names=['paths', 'class'])
        root_folder = df['paths'].head(1)[0]
        df = df.tail(df.shape[0] - 1)
        df.drop_duplicates()
        df['class'] = df['class'].astype(int)
        df = df.sort_values('class')

        order_list = [i for i in range(1 + max(list(set(df['class'].values.tolist()))))]
        order_list_reverse = [order_list.index(i) for i in list(set(df['class'].values.tolist()))]
        samples = [(os.path.join(root_folder, elt[0]), order_list_reverse[elt[1]]) for elt in
                   list(map(tuple, df.values.tolist()))]
        samples.sort(key=lambda x: x[1])
        if not samples:
            raise (RuntimeError("No image found"))

        # 设置数据加载器，默认为pil_loader函数
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
        self.indices = indices
        self.return_item_ix = return_item_ix

    # 获取数据集中指定索引处的样本
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # 如果设置了返回项索引，则返回样本、目标和索引；否则，仅返回样本和目标
        if self.return_item_ix:
            return sample, target, index
        else:
            return sample, target

    # 获取数据集的长度
    def __len__(self):
        return len(self.indices)

    # 返回数据集的字符串表示形式，包括数据点数量、根目录位置以及转换和目标转换信息（如果有）
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class DataUtils():
    # 初始化函数
    def __init__(self):
        return

    # 计算模型预测准确率
    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    # 从数据集均值和标准差文件中获取指定数据集的均值和标准差
    def get_dataset_mean_std(self, normalization_dataset_name, datasets_mean_std_file_path):
        datasets_mean_std_file = open(datasets_mean_std_file_path, 'r').readlines()
        for line in datasets_mean_std_file:
            line = line.strip().split(':')
            dataset_name, dataset_stat = line[0], line[1]
            if dataset_name == normalization_dataset_name:
                dataset_stat = dataset_stat.split(';')
                dataset_mean = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[0])]
                dataset_std = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[1])]
                return dataset_mean, dataset_std
        print('Invalid normalization dataset name')
        sys.exit(-1)

        # 将字符串转换为指定类型的列表
    def from_str_to_list(self, string, type):
        list = []
        params = string.split(',')
        for p in params:
            if type == 'int':
                list.append(int(p.strip()))
            elif type == 'float':
                list.append(float(p.strip()))
            elif type == 'str':
                list.append(str(p.strip()))
        return list

# 创建一个3x3的卷积层
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        # 初始化基础块，包括卷积层、批归一化、ReLU激活函数等配置
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        # 定义基础块的前向传播逻辑
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if not self.last:
            out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        # 初始化 ResNet 模型，包括卷积层、批归一化、全连接层等配置
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, last_phase=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = modified_linear.CosineLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        # 创建 ResNet 中的每个层，包括堆叠 block 函数生成的块
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 定义 ResNet 的前向传播逻辑
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

def resnet18(pretrained=False, **kwargs):
    # 初始化一个ResNet18模型
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def merge_images_labels(images, labels):
    # 合并图像和标签为包含元组的列表
    images = list(images)
    labels = list(labels)
    assert (len(images) == len(labels))
    imgs = []
    for i in range(len(images)):
        item = (images[i], labels[i])
        imgs.append(item)
    return imgs

if __name__ == '__main__':
    # 主程序入口
    if len(sys.argv) != 2:
        # 检查命令行参数
        print('Arguments: config')
        sys.exit(-1)

    # 读取配置文件
    cp = ConfigParser()
    with open(sys.argv[1]) as fh:
        cp.read_file(fh)

    cp = cp['config']
    # 读取配置中的参数
    nb_classes = int(cp['nb_classes'])
    normalization_dataset_name = cp['dataset']
    first_batch_size = int(cp["first_batch_size"])
    il_states = int(cp["il_states"])
    feat_root = cp["feat_root"]
    list_root = cp["list_root"]
    model_root = cp["model_root"]
    random_seed = int(cp["random_seed"])
    num_workers = int(cp['num_workers'])
    epochs_lucir = int(cp['epochs_lucir'])
    epochs_augmix_ft = int(cp['epochs_augmix_ft'])

    print(normalization_dataset_name)

    B = first_batch_size
    # 设置输出路径
    datasets_mean_std_file_path = cp["mean_std"]
    output_dir = os.path.join(model_root, normalization_dataset_name, "seed" + str(random_seed),
                              "b" + str(first_batch_size))
    train_file_path = os.path.join(list_root, normalization_dataset_name, "train.lst")
    test_file_path = os.path.join(list_root, normalization_dataset_name, "test.lst")

    # 初始化DataUtils
    utils = DataUtils()
    train_batch_size = 128
    test_batch_size = 50
    eval_batch_size = 128
    base_lr = 0.1
    lr_strat = [30, 60]
    lr_factor = 0.1
    custom_weight_decay = 0.0001
    custom_momentum = 0.9

    epochs = epochs_lucir

    # 打印当前设备信息
    print("Running on " + str(socket.gethostname()) + " | " + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '\n')
    # 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取数据集的均值和标准差
    dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)

    # 设置数据预处理
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    top = min(5, B)

    # 构建训练和测试数据集
    trainset = ImagesListFileFolder(
        train_file_path,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), random_seed=random_seed, range_classes=range(B))

    testset = ImagesListFileFolder(
        test_file_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), random_seed=random_seed, range_classes=range(B))

    # 获取训练和测试数据
    X_train_total, Y_train_total = np.array(trainset.imgs), np.array(trainset.targets)
    X_valid_total, Y_valid_total = np.array(testset.imgs), np.array(testset.targets)

    order_list = list(range(B))

    # 存储训练和验证数据
    X_valid_cumuls = []
    X_protoset_cumuls = []
    X_train_cumuls = []
    Y_valid_cumuls = []
    Y_protoset_cumuls = []
    Y_train_cumuls = []

    # 初始化ResNet18模型并获取其输入输出特征
    tg_model = resnet18(num_classes=B)
    in_features = tg_model.fc.in_features
    out_features = tg_model.fc.out_features
    print("in_features:", in_features, "out_features:", out_features)

    # 设置训练和验证数据
    X_train = X_train_total
    X_valid = X_valid_total
    X_valid_cumuls.append(X_valid)
    X_train_cumuls.append(X_train)
    X_valid_cumul = np.concatenate(X_valid_cumuls)
    X_train_cumul = np.concatenate(X_train_cumuls)
    Y_train = Y_train_total
    Y_valid = Y_valid_total
    Y_valid_cumuls.append(Y_valid)
    Y_train_cumuls.append(Y_train)
    Y_valid_cumul = np.concatenate(Y_valid_cumuls)
    Y_train_cumul = np.concatenate(Y_train_cumuls)
    X_valid_ori = X_valid
    Y_valid_ori = Y_valid
    # 映射训练和验证标签
    map_Y_train = np.array([order_list.index(i) for i in Y_train])
    map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])

    # 合并图像和标签
    current_train_imgs = merge_images_labels(X_train, map_Y_train)
    trainset.imgs = trainset.samples = current_train_imgs
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=True)

    print('Training-set size = ' + str(len(trainset)))

    current_test_imgs = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
    testset.imgs = testset.samples = current_test_imgs
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=num_workers)

    print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
    print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))

    # 检查点文件名
    ckp_name = os.path.join(output_dir, 'lucir_scratch.pth')
    print('ckp_name', ckp_name)

    # 模型参数和优化器
    tg_params = tg_model.parameters()

    tg_model = tg_model.to(device)
    tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
    tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)

    # 训练模型
    for epoch in range(epochs):
        tg_model.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            targets = targets.long()
            outputs = tg_model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            tg_optimizer.step()
        tg_lr_scheduler.step()

        top1 = AverageMeter()
        top5 = AverageMeter()
        tg_model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, top))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

        print('{:03}/{:03} | Test ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(
            epoch + 1, epochs, len(testloader), top1.avg, top, top5.avg))

    # 保存模型
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    torch.save(tg_model.state_dict(), ckp_name)
    ckp_name = os.path.join(output_dir, 'lucir_scratch.pth')
    epochs = epochs_augmix_ft
    batch_size = 64
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0001
    lrd = 10

    #计算top准确率
    def accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    class Lookahead(Optimizer):

        def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
            # 初始化 Lookahead 优化器及其参数
            self.optimizer = optimizer
            self._la_step = 0
            self.la_alpha = la_alpha
            self._total_la_steps = la_steps
            pullback_momentum = pullback_momentum.lower()
            assert pullback_momentum in ["reset", "pullback", "none"]
            self.pullback_momentum = pullback_momentum

            self.state = defaultdict(dict)

            for group in optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['cached_params'] = torch.zeros_like(p.data)
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        param_state['cached_mom'] = torch.zeros_like(p.data)

        def __getstate__(self):
            # 返回对象的状态字典
            return {
                'state': self.state,
                'optimizer': self.optimizer,
                'la_alpha': self.la_alpha,
                '_la_step': self._la_step,
                '_total_la_steps': self._total_la_steps,
                'pullback_momentum': self.pullback_momentum
            }

        def zero_grad(self):
            # 将梯度清零
            self.optimizer.zero_grad()

        def get_la_step(self):
            # 返回当前 Lookahead 步数
            return self._la_step

        def state_dict(self):
            # 返回优化器的状态字典
            return self.optimizer.state_dict()

        def load_state_dict(self, state_dict):
            # 加载优化器的状态字典
            self.optimizer.load_state_dict(state_dict)

        def _backup_and_load_cache(self):
            # 备份当前参数并加载缓存的参数
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['backup_params'] = torch.zeros_like(p.data)
                    param_state['backup_params'].copy_(p.data)
                    p.data.copy_(param_state['cached_params'])

        def _clear_and_load_backup(self):
            # 清除缓存并加载备份的参数
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.copy_(param_state['backup_params'])
                    del param_state['backup_params']

        @property
        def param_groups(self):
            # 返回优化器的参数组
            return self.optimizer.param_groups

        def step(self, closure=None):
            # 执行 Lookahead 优化算法
            loss = self.optimizer.step(closure)
            self._la_step += 1

            if self._la_step >= self._total_la_steps:
                self._la_step = 0
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        param_state = self.state[p]
                        p.data.mul_(self.la_alpha).add_(param_state['cached_params'],
                                                        alpha=1.0 - self.la_alpha)  # 进行参数更新
                        param_state['cached_params'].copy_(p.data)
                        if self.pullback_momentum == "pullback":
                            # 如果设定了动量回撤，则更新动量缓冲区
                            internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                            self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                                1.0 - self.la_alpha, param_state["cached_mom"])
                            param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                        elif self.pullback_momentum == "reset":
                            # 如果设定了重置动量，则将动量缓冲区清零
                            self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

            return loss


    # 判断是否存在 GPU 设备，如果存在则打印使用 GPU 设备信息
    if device is not None:
        print("Use GPU: {} for training".format(device))

    # 加载 ResNet-18 模型结构
    model = models.resnet18()

    # 从检查点文件加载模型参数
    tg_model_state_dict = torch.load(ckp_name)
    print("Loading lucir_model from {}".format(ckp_name))
    state_dict = tg_model_state_dict

    # 删除状态字典中以 'fc' 开头的键
    for key in list(state_dict.keys()):
        if key.startswith('fc'):
            del state_dict[key]

    # 修改模型的最后一层全连接层
    model.fc = nn.Linear(512, B)

    # 载入模型参数，不严格匹配
    model.load_state_dict(state_dict, strict=False)

    # 打印模型已加载的提示
    print('modele charge')

    # 将模型迁移到指定的 GPU 上
    model.cuda(device)

    # 定义损失函数和优化器，并设定在 GPU 上执行
    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    # 包装优化器为 Lookahead 优化器
    optimizer = Lookahead(optimizer)

    # 创建训练数据集，包含数据增强和预处理过程
    train_dataset = ImagesListFileFolder(
        train_file_path,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.AugMix(severity=5, chain_depth=7),
            transforms.ToTensor(),
            normalize,
        ]), random_seed=random_seed, range_classes=range(B))

    # 创建验证数据集，包含预处理过程
    val_dataset = ImagesListFileFolder(
        test_file_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), random_seed=random_seed, range_classes=range(B))

    # 创建数据加载器用于训练集和验证集
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                               pin_memory=False, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                             pin_memory=False, shuffle=False)

    # 打印数据集信息
    print('Classes number = {}'.format(len(train_dataset.classes)))
    print('Training dataset size = {}'.format(len(train_dataset)))
    print('Validation dataset size = {}'.format(len(val_dataset)))


    # 定义调整学习率的函数
    def adjust_learning_rate(optimizer, epoch, lr):
        if epoch == 80 or epoch == 120:
            lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # 打印训练开始的提示
    print('\nstarting training...')
    start = time.time()

    # 开始训练过程
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # 切换模型到训练模式
        model.train()
        for i, (input, target) in enumerate(train_loader):
            if device is not None:
                input = input.cuda(device)
            target = target.cuda(device)
            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end = time.time()
        epoch_time = timedelta(seconds=round(end - start))

        # 打印每个 epoch 的训练结果
        print('{:03}/{:03} | Train ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f} | loss = {:.4f}'.format(
            epoch + 1, epochs, len(train_loader), top1.avg, top, top5.avg, losses.avg))
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # 切换模型到评估模式并进行验证
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, top))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
        print('        | Test  ({})'.format(len(val_loader)) + ' ' * (len(str(len(train_loader))) - len(
            str(len(val_loader)))) + ' |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(top1.avg, top, top5.avg))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

    # 保存训练的模型
    ckp_name = os.path.join(output_dir, 'scratch.pth')
    torch.save(model, ckp_name)


    # 定义特征提取函数，将特征模型、数据加载器、根路径和 GPU 设备作为输入参数
    def features_extraction(features_model, loader, root_path, gpu):
        # 将特征提取模型迁移到指定的 GPU 上
        features_model = features_model.cuda(gpu)
        # 设置模型为评估模式
        features_model.eval()
        try:
            # 删除特征存储目录的旧文件
            print('cleaning', root_path, '...')
            shutil.rmtree(root_path)
        except:
            # 如果目录不存在，忽略错误
            pass
        # 创建特征存储目录
        os.makedirs(root_path, exist_ok=True)
        last_class = -1
        # 遍历数据加载器中的数据
        for i, (inputs, labels) in enumerate(loader):
            # 将输入数据迁移到 GPU 上
            inputs = inputs.cuda(gpu)
            # 提取特征
            features = features_model(inputs)
            # 获取标签和特征的列表
            lablist = labels.tolist()
            featlist = features.tolist()
            # 保存提取的特征到对应的文件中
            for i in range(len(lablist)):
                cu_class = lablist[i]
                if cu_class != last_class:
                    last_class = cu_class
                # 以追加模式写入文件
                with open(os.path.join(root_path, str(cu_class)), 'a') as features_out:
                    features_out.write(str(' '.join([str(e[0][0]) for e in list(featlist[i])])) + '\n')


    # 创建训练数据集，包含预处理过程
    train_dataset = ImagesListFileFolder(
        train_file_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), random_seed=random_seed, range_classes=range(nb_classes))

    # 创建验证数据集，包含预处理过程
    val_dataset = ImagesListFileFolder(
        test_file_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), random_seed=random_seed, range_classes=range(nb_classes))

    # 创建数据加载器用于训练集和验证集，批次大小为1
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=num_workers, pin_memory=False,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=num_workers, pin_memory=False,
                                             shuffle=False)

    # 定义训练特征存储路径
    train_features_path = os.path.join(output_dir, 'train')

    # 拼接特征存储路径的目录结构
    feat_dir_full = os.path.join(feat_root, normalization_dataset_name, "seed" + str(random_seed),
                                 "b" + str(first_batch_size))
    train_features_path = os.path.join(feat_dir_full, 'train')
    val_features_path = os.path.join(feat_dir_full, 'test')

    # 提取模型中的特征层
    feat_model = nn.Sequential(*list(model.children())[:-1])

    # 提取训练数据集的特征并保存
    features_extraction(feat_model, train_loader, train_features_path, device)

    # 提取验证数据集的特征并保存
    features_extraction(feat_model, val_loader, val_features_path, device)