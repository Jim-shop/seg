import os
from typing import Tuple

import cv2
import numpy as np
import webcolors


def GetConfusionMatrix(numClass: int, imgPredict: np.ndarray, Label: np.ndarray) -> np.ndarray:
    """返回混淆矩阵"""
    mask = (Label >= 0) & (Label < numClass)  # 仅处理合法数据
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass**2)
    ConfusionMatrix = count.reshape(numClass, numClass)
    return ConfusionMatrix


def GetAccuracyInfo(PredictPath: str, LabelPath: str, ImgSize: int, SavePath: str) -> Tuple[np.ndarray, np.ndarray, np.float64, np.ndarray, np.float64, np.ndarray]:
    """
    取得准确率信息
    返回：
    precision, recall, oA, IoU, mIoU, F1Score
    """
    from data.ColorDict import colorDictBGR
    colorDictGray = colorDictBGR.reshape(
        (colorDictBGR.shape[0], 1, colorDictBGR.shape[1])).astype(np.uint8)
    colorDictGray = cv2.cvtColor(colorDictGray, cv2.COLOR_BGR2GRAY).squeeze()
    # 获取文件夹内所有图像
    PredictList = os.listdir(PredictPath)
    # 图像大小
    imgSize = (ImgSize, ImgSize)
    # 图像数目
    label_num = len(PredictList)
    # 把所有图像放在一个数组里
    label_all = np.zeros((label_num,) + imgSize, np.uint8)
    predict_all = np.zeros((label_num,) + imgSize, np.uint8)
    for i in range(label_num):
        label = cv2.imread(os.path.join(LabelPath, PredictList[i]),
                           cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, imgSize,
                           interpolation=cv2.INTER_NEAREST)
        label_all[i] = label
        predict_all[i] = cv2.imread(os.path.join(
            PredictPath, PredictList[i]), cv2.IMREAD_GRAYSCALE)
    # 把颜色映射到0,1,2,3
    for i in range(colorDictGray.shape[0]):
        label_all[label_all == colorDictGray[i]] = i
        predict_all[predict_all == colorDictGray[i]] = i
    # 拉直成一维
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()
    # 计算混淆矩阵及各精度参数
    """
    0 \ 1 | 实际真 | 实际假
    ——————————————————————
    预测真 |   TP     FP
    预测假 |   FN     TN

    TP = True Postive = 真阳性； FP = False Positive = 假阳性
    FN = False Negative = 假阴性； TN = True Negative = 真阴性

    精度 Precision           = TP / (TP + FP)
    召回 Recall              = TP / (TP + FN)
    特异度 Specificity        = TN / (TN + FP)
    总体精度 Overall Accuracy = (TP + TN) / (TP + TN + FP + FN)
    交并比 IoU                = (TP + TN) / (TP + FP + FN)
    平均交并比 mIoU           = 每一类IoU的平均
    频权交并比 FWIoU          = 每一类IoU实际频率为权重的加权平均值
    F1分数 F1Score           = Precision和Recall的调和平均数
    """
    confusionMatrix = GetConfusionMatrix(
        colorDictBGR.shape[0], predict_all, label_all)

    sum = confusionMatrix.sum()
    sum0axis = confusionMatrix.sum(axis=0)
    sum1axis = confusionMatrix.sum(axis=1)
    intersection = np.diag(confusionMatrix)
    union = sum0axis + sum1axis - intersection
    freq = sum1axis / sum

    precision = intersection / sum0axis  # 每一类
    recall = intersection / sum1axis  # 每一类
    oA = intersection.sum() / sum
    IoU = intersection / union  # 每一类
    mIoU = np.nanmean(IoU)
    FWIoU = (freq[freq > 0] * IoU[freq > 0]).sum()
    F1Score = 2 * precision * recall / (precision + recall)

    msg = (
        f"【准确率数据】\n"
        f"混淆矩阵：\n{confusionMatrix}\n"
        f"精度：\n{precision}\n"
        f"召回：\n{recall}\n"
        f"总体准确率：\n{oA}\n"
        f"IoU：\n{IoU}\n"
        f"mIoU:\n{mIoU}\n"
        f"FWIoU：\n{FWIoU}\n"
        f"F1-Score：\n{F1Score}\n"
    )
    print("\n"+msg+"\n")
    with open(f"{SavePath}/accuracy.txt", "w") as f:
        f.writelines(msg)

    return precision, recall, oA, IoU, mIoU, F1Score
