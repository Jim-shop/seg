from random import random
from typing import Tuple

import cv2
import numpy as np
import torch

from tools.dataloader import SegDataType


class Resize(object):
    """
    缩放图像和标签到指定大小
    """

    def __init__(self, size: int, interpolation: int = None):
        self.size = (size, size)
        self.flagvals = interpolation

    def __call__(self, sample: SegDataType) -> SegDataType:
        for elem in sample.keys():
            # 默认灰度图是最邻近，彩色图是立方插值
            if self.flagvals is None:
                if sample[elem].ndim == 2:
                    flagval = cv2.INTER_NEAREST
                else:
                    flagval = cv2.INTER_CUBIC
            # 如果是灰度图/三色彩图
            if sample[elem].ndim == 2 or (sample[elem].ndim == 3 and sample[elem].shape[2] == 3):
                sample[elem] = cv2.resize(
                    sample[elem], self.size, interpolation=flagval)
            # 如果不是上面的类型
            else:
                # TODO
                print("这里什么鬼")
                exit(0)
                tmp = np.array(sample)
                sample[elem] = np.zeros(
                    np.append(self.size, tmp.shape[2]), dtype=np.float32)
                for ii in range(sample[elem].shape[2]):
                    sample[elem][:, :, ii] = cv2.resize(
                        tmp[:, :, ii], self.size, interpolation=flagval)
        return sample

    def __str__(self):
        return f"Resize {str(self.size)}"


class RandomHorizontalFlip(object):
    """
    以0.5的概率水平翻转图像和标签
    """

    def __call__(self, sample: SegDataType) -> SegDataType:
        if random() < 0.5:
            for elem in sample.keys():
                sample[elem] = cv2.flip(sample[elem], 1)
        return sample

    def __str__(self):
        return "RandomHorizontalFlip"


class RandomVerticalFlip(object):
    """
    以概率0.5的概率垂直翻转图像和标签
    """

    def __call__(self, sample: SegDataType) -> SegDataType:
        if random() < 0.5:
            for elem in sample.keys():
                sample[elem] = cv2.flip(sample[elem], 0)
        return sample

    def __str__(self):
        return "RandomVerticalFlip"


class RandomScaleAndRotate(object):
    """
    随机缩放和旋转图像和标签。
    """

    def __init__(self, rots: Tuple[int, int] = (-30, 30), scales: Tuple[int, int] = (.75, 1.25)):
        """用元组给定区间"""
        self.rots = rots
        self.scales = scales

    def __call__(self, sample: SegDataType) -> SegDataType:
        rot = (self.rots[1] - self.rots[0]) * random() + self.rots[0]
        sc = (self.scales[1] - self.scales[0]) * random() + self.scales[0]
        for elem in sample.keys():
            if sample[elem].ndim == 2:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            h, w = sample[elem].shape[:2]
            sample[elem] = cv2.warpAffine(
                sample[elem],
                cv2.getRotationMatrix2D((w/2, h/2), rot, sc),
                (w, h), flags=flagval)
        return sample

    def __str__(self):
        return "RandomScaleAndRotate"


class Normalize(object):
    """
    正则化图像标签组中的图像
    """

    def __init__(self, mean: Tuple[float, float, float] = (0., 0., 0.), std: Tuple[float, float, float] = (1., 1., 1.,)):
        self.mean = mean
        self.std = std

    def __call__(self, sample: SegDataType) -> SegDataType:
        sample["image"] = sample["image"] / 255.
        sample["image"] -= self.mean
        sample["image"] /= self.std
        return sample

    def __str__(self):
        return "Normalize"


class ToTensor(object):
    """
    将图像和标签ndarray组转成tensor
    """

    def __call__(self, sample: SegDataType) -> SegDataType:
        for elem in sample.keys():
            sample[elem] = sample[elem].astype(np.float32)

            if sample[elem].ndim == 2:
                sample[elem] = sample[elem][:, :, np.newaxis]

            # 交换颜色所在维度，因为：
            # numpy结构: H x W x C
            # torch结构: C x H x W
            sample[elem] = sample[elem].transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(sample[elem]).float()
        return sample

    def __str__(self):
        return "ToTensor"
