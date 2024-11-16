import os
import math
import argparse
import sys

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
from model.CV_Model_Config.resnet import resnet50 as create_model
from model.CV_Model_Config.my_dataset import MyDataSet
from model.CV_Model_Config.utils import read_split_data, train_one_epoch, evaluate

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    os.makedirs("./model/CV_Models/trained_model", exist_ok=True)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

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

    # 实例化数据集
    train_dataset = MyDataSet(train_images_path, train_images_label, transform=data_transform["train"])
    val_dataset = MyDataSet(val_images_path, val_images_label, transform=data_transform["val"])

    nw = min([os.cpu_count(), args.batch_size, 8])
    print(f'Using {nw} dataloader workers per process.')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn
    )

    # 创建模型并加载预训练权重
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights and os.path.exists(args.weights):
        weights = torch.load(args.weights, map_location='cpu')
        model.load_state_dict({k: v for k, v in weights.items() if k in model.state_dict() and 'fc' not in k}, strict=False)
        print(f'Loaded weights from {args.weights}')



    # 冻结层（如果指定）
    if args.freeze_layers:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    # 优化器和学习率调度器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 训练循环
    for epoch in range(args.epochs):
     mean_loss, accuracy = train_one_epoch(
        model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch
     )

    # 从优化器中提取当前学习率
     current_lr = optimizer.param_groups[0]["lr"]

     val_loss, val_accuracy = evaluate(
        model=model, data_loader=val_loader, device=device, epoch=epoch
     )

     print(f"[epoch {epoch}] val_loss: {val_loss:.3f}, val_accuracy: {val_accuracy:.3f}")

     tags = ["loss", "accuracy", "learning_rate"]
     tb_writer.add_scalar(tags[0], mean_loss, epoch)
     tb_writer.add_scalar(tags[1], accuracy, epoch)
     tb_writer.add_scalar(tags[2], current_lr, epoch)
     best_acc=0.0
     if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), f"./model/CV_Models/trained_model/resnet50.pth")

     scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=61)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default="./data/CV/CV_dataset/train-val")
    parser.add_argument('--weights', type=str, default='./model/CV_Models/pretrain_model/resnet34-pre.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
