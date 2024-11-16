import os
import sys
import json
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from model.CV_Model_Config.utils import read_split_data, train_one_epoch, evaluate
from model.CV_Model_Config.FasterNett0 import FasterNet
from model.CV_Model_Config.my_dataset import MyDataSet

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    os.makedirs("./model/CV_Models/trained_model", exist_ok=True)
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    # 数据增强与预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 加载数据集
    train_dataset = MyDataSet(train_images_path, train_images_label, transform=data_transform["train"])
    val_dataset = MyDataSet(val_images_path, val_images_label, transform=data_transform["val"])
    nw = min([os.cpu_count(), args.batch_size, 8])
    print(f'Using {nw} dataloader workers per process.')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        pin_memory=True, num_workers=nw
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        pin_memory=True, num_workers=nw
    )

    print(f"Using {len(train_dataset)} images for training, {len(val_dataset)} images for validation.")

    # 创建模型
    model = FasterNet(num_classes=args.num_classes).to(device)

    # 加载预训练权重
    if args.weights and os.path.exists(args.weights):
        pre_weights = torch.load(args.weights, map_location='cpu')
        # 只加载与模型结构匹配的权重
        pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(pre_dict, strict=False)
        print(f'Loaded weights from {args.weights}')

    # 冻结特征层权重（可选）
    if args.freeze_layers:
        for param in model.features.parameters():
            param.requires_grad = False

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    # 定义学习率调度器
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, (images, labels) in enumerate(train_bar):
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{args.epochs}] loss:{loss:.3f}"

        # 验证阶段
        model.eval()
        acc = 0.0  # 累积准确率
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_images, val_labels in val_bar:
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = f"valid epoch[{epoch + 1}/{args.epochs}]"

        val_accuracy = acc / len(val_dataset)
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  val_accuracy: {val_accuracy:.3f}')

        # TensorBoard记录
        tb_writer.add_scalar("loss", running_loss / len(train_loader), epoch)
        tb_writer.add_scalar("accuracy", val_accuracy, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # 保存最优模型权重
        # 保存最优模型权重
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), f"./model/CV_Models/trained_model/fasternett1.pth")

        scheduler.step()

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=61)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default="./data/CV/CV_dataset/train-val")
    parser.add_argument('--weights', type=str, default='./model/CV_Models/pretrain_model/fasternet_t1.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
