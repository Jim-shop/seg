import os
import random

if __name__ == "__main__":

    di = os.path.abspath(r".\ProcessData\images")
    dl = os.path.abspath(r".\ProcessData\labels")

    images = os.listdir(di)
    random.shuffle(images)

    train_ratio = 0.9
    # 验证集复用训练集
    # 其余属于测试集

    train_count_pos = int(train_ratio*len(images))

    with open("train_list.txt", "w") as f:
        for name in images[:train_count_pos]:
            f.writelines(
                f"{os.path.join(di,name)}  {os.path.join(dl,name)}\n")
    with open("val_list.txt", "w") as f:
        for name in images[:train_count_pos]:  # 验证集与训练集相同
            f.writelines(
                f"{os.path.join(di,name)}  {os.path.join(dl,name)}\n")
    with open("test_list.txt", "w") as f:
        for name in images[train_count_pos:]:
            f.writelines(
                f"{os.path.join(di,name)}  {os.path.join(dl,name)}\n")
