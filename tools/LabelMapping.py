import numpy as np


def LabelMapping(label: np.ndarray) -> np.ndarray:
    colorTable = np.array([[0, 0, 0],  # bg
                           [255, 0, 0],  # 1
                           [0, 255, 0],  # 2
                           [0, 0, 255],  # 3
                           ])
    return colorTable[label, :].reshape([label.shape[0], label.shape[1], 3])
