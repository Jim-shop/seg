import os.path

import cv2
import numpy as np
from tqdm import tqdm

from ColorDict import colorDictBGR

GT_ICM = "BlastsOnline/GT_ICM"
GT_TE = "BlastsOnline/GT_TE"
GT_ZP = "BlastsOnline/GT_ZP"
Images = "BlastsOnline/Images"

save_labels = "ProcessData/labels"
save_visual = "ProcessData/visual"

names = os.listdir(Images)

if __name__ == "__main__":
    for item in tqdm(names):
        icm = cv2.imread(os.path.join(GT_ICM, item.replace(".BMP", " ICM_Mask.bmp")), cv2.IMREAD_GRAYSCALE)
        te = cv2.imread(os.path.join(GT_TE, item.replace(".BMP", " TE_Mask.bmp")), cv2.IMREAD_GRAYSCALE)
        zp = cv2.imread(os.path.join(GT_ZP, item.replace(".BMP", " zp_Mask.bmp")), cv2.IMREAD_GRAYSCALE)
        size = icm.shape
        label = np.zeros(size, dtype="uint8")
        label = np.where(icm > 0, 1, label)
        label = np.where(te > 0, 2, label)
        label = np.where(zp > 0, 3, label)
        cv2.imwrite(os.path.join(save_labels, item), label)
        visual = colorDictBGR[label]
        cv2.imwrite(os.path.join(save_visual, item), visual)
