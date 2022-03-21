from typing import Tuple, Callable
from torch import nn, Tensor


def GetLossFn(LossFnName: str, ClassWeight: Tuple[Tensor, None] = None) -> Callable:
    # "ce","DiceLoss","FocalLoss","CE_DiceLoss","LovaszSoftmax","bce"
    if LossFnName == "ce":
        return nn.CrossEntropyLoss(weight=ClassWeight, ignore_index=255)
