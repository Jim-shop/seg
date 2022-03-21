import os
from typing import Union, Dict

import cv2
import numpy
from torch.utils.data import Dataset

"""
PyTorch自定义数据集必须重载getitem()和len()
"""

SegDataType = Dict[str, Union[numpy.ndarray, str]]


class Segmentation(Dataset):
    """
    胚胎分割的数据集。返回一个dict，内有三个索引，
    - "image": ndarray -> 胚胎原始图像（RGB）
    - "gt": ndarray -> 标签（三色）
    - "name": str -> 文件名
    """

    def __init__(self, txtPath: str = None, transform=None):
        """ 
        输入：
        txtPath: str -> 为 DataGrouping.py 所生成的一个.txt文件路径，
        transform -> 对数据项的变换
        """
        super().__init__()
        self.list_sample = open(txtPath, "r").readlines()  # 执行完后临时变量销毁，文件自动关闭
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0, "读取不到数据集"
        self.transform = transform

    def __len__(self) -> int:
        return self.num_sample

    def __getitem__(self, index: int) -> SegDataType:
        image_path, label_path = self.list_sample[index].strip().split("  ")
        sample = {"image": cv2.imread(image_path, cv2.IMREAD_COLOR),
                  "gt": cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)}
        if self.transform is not None:
            sample = self.transform(sample)
        sample["name"] = os.path.basename(label_path)
        return sample
