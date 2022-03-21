import os.path
import sys
from typing import Callable, Tuple

import cv2
import numpy as np
import torch
import torchvision
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

import tools.transform as tr
from network.GetModel import GetNet
from tools.dataloader import Segmentation
from tools.FindNewFile import FindNewFile
from tools.LabelMapping import LabelMapping
from tools.loss import GetLossFn
from tools.ParseYAML import ParseYAML
from tools.Metrics import GetAccuracyInfo


def eval(val_loader: DataLoader, model: torch.nn.Module, criterion: Callable, epoch: int) -> Tuple[float, float, float, float]:
    """
    模型评估。
    """
    model.eval()  # 切换模型到评估模式，改变部分层行为
    if params["val_visual"]:
        val_visual_path = os.path.join(params["save_dir_model"], "val_visual")
        val_visual_epoch_path = os.path.join(val_visual_path, str(epoch))
        if os.path.exists(val_visual_path) == False:
            os.mkdir(val_visual_path)
        if os.path.exists(val_visual_epoch_path) == False:
            os.mkdir(val_visual_epoch_path)
    with torch.no_grad():
        batch_num = 0
        val_loss = 0.0
        for _, data in tqdm(enumerate(val_loader), desc="评估进度", total=len(val_data)):
            images, labels, names = data["image"], data["gt"], data["name"]
            labels = labels.view(
                images.size()[0], params["img_size"], params["img_size"]).long()
            if params["use_gpu"]:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            val_losses = criterion(outputs, labels)
            outputs = torch.argmax(outputs, 1)
            pred = outputs.cpu().data.numpy().astype(np.int32)
            batch_num += images.size()[0]
            val_loss += val_losses.item()
            if params["val_visual"]:
                for kk in range(len(names)):
                    vis = LabelMapping(pred[kk, :, :])
                    cv2.imwrite(os.path.join(
                        val_visual_epoch_path, names[kk]), vis)
            if _ >= 1:  # 测试用 TODO
                break
        _, _, oA, IoU, _, F1Score = GetAccuracyInfo(
            val_visual_epoch_path,
            params["val_gt"],
            params["num_class"],
            val_visual_epoch_path
        )
        val_loss = val_loss/batch_num
    return IoU[1], oA, F1Score[1], val_loss  # 只评估一个维度


if __name__ == "__main__":
    """
    命令行:
    python train.py ["配置文件.yaml"]
    """
    # 读取配置文件
    if len(sys.argv) == 1:
        yaml_file = "config.yaml"
    else:
        yaml_file = sys.argv[1]
    params = ParseYAML(yaml_file)
    # 解析GPU参数
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    gpu_list = list(set(map(int, params["gpu_id"].split(","))))  # 分割、去重
    if len(gpu_list) < 2:
        params["use_gpu"] = False
    # 构造数据集
    train_transforms = torchvision.transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomVerticalFlip(),
        tr.RandomScaleAndRotate(rots=(-15, 15), scales=(0.9, 1.1)),
        tr.Resize(params["img_size"]),
        tr.Normalize(mean=params["mean"], std=params["std"]),
        tr.ToTensor()
    ])
    val_trainsforms = torchvision.transforms.Compose([
        tr.Resize(params["img_size"]),
        tr.Normalize(mean=params["mean"], std=params["std"]),
        tr.ToTensor()
    ])
    train_data = Segmentation(txtPath=params["train_list"],
                              transform=train_transforms)
    train_loader = DataLoader(train_data,
                              batch_size=params["batch_size"],
                              shuffle=True,
                              num_workers=params["num_workers"],
                              drop_last=True)
    val_data = Segmentation(txtPath=params["val_list"],
                            transform=val_trainsforms)
    val_loader = DataLoader(val_data,
                            batch_size=params["batch_size"],
                            shuffle=False,
                            num_workers=params["num_workers"],
                            drop_last=True)
    # 定义模型
    model = GetNet(model_name=params["model_name"],
                   in_channel=params["input_bands"],
                   num_class=params["num_class"],
                   pretrained_path=params["pretrained_model"])
    if params["use_gpu"]:
        model = torch.nn.DataParallel(model, device_ids=gpu_list)
    last_state = FindNewFile(params["model_dir"])    # 载入上一次模型
    if last_state is not None:
        model.load_state_dict(torch.load(last_state))
        print(f"载入模型{last_state}")
    if params["use_gpu"]:
        model.cuda()
    # 损失函数
    criterion = GetLossFn(params["loss_type"])
    if params["extra_loss"]:
        weights_f = torch.tensor([0.01, 1]).float()
        weights_b = torch.tensor([1, 0.01]).float()
        criterion_f = GetLossFn("ce", ClassWeight=weights_f)
        criterion_b = GetLossFn("ce", ClassWeight=weights_b)
    # 优化器
    optimizer = create_optimizer_v2(model, "adam", lr=params["base_lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.8)

    #############
    ## 开始训练 ##
    #############

    best_val_acc = .0
    with open(os.path.join(params["save_dir_model"], "log.txt"), "w") as log:
        for epoch in range(params["epoches"]):
            model.train()  # 进入训练模式（而不是评估模式）
            running_loss = .0
            batch_num = 0
            for i, data in tqdm(enumerate(train_loader), desc="训练进度", total=len(train_data)):
                images, labels = data["image"], data["gt"]
                # 数据结构：torch.Size([batch_size, C, H, W])
                # i += images.size()[0]
                labels = labels.squeeze().long()
                if params["use_gpu"]:
                    images = images.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()
                if params["extra_loss"]:
                    outputs, output_f, output_b = model(images)
                    losses1 = criterion(outputs, labels)
                    losses_f = criterion_f(output_f, labels)
                    losses_b = criterion_b(output_b, labels)
                    losses = losses1 + losses_f * .7 + losses_b * .3
                else:
                    outputs = model(images)
                    losses = criterion(outputs, labels)
                losses.backward()
                optimizer.step()
                running_loss += losses
                batch_num += images.size()[0]
                if i >= 1:  # 测试用 TODO
                    break

            print(f"\n轮数: {epoch}\n"
                  f"训练损失: {running_loss.item()/batch_num}\n")
            scheduler.step()

            if epoch % params["save_iter"] == 0:  # 保存
                val_miou, val_acc, val_f1, val_loss = eval(
                    val_loader, model, criterion, epoch)
