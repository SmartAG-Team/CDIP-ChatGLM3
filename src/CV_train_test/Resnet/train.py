import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
import time

from model import resnet101

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = r"..\\train"

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])
    print('Using {} dataloader workers for each process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("Using {} images for training, {} images for validation.".format(train_num, val_num))

    net = resnet101()
    model_weight_path = "./resnet101-pre.pth"
    assert os.path.exists(model_weight_path), "File {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 52)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.00001)

    epochs = 15
    best_acc = 0.0
    save_path = './resNet101.pth'
    train_steps = len(train_loader)

    train_loss_list = []
    val_accuracy_list = []
    val_recall_list = []
    val_precision_list = []
    val_f1_list = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        start_time = time.time()  # 记录开始时间
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "Train Epoch [{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        fps = train_steps / elapsed_time  # 计算FPS

        net.eval()
        acc = 0.0
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                true_labels.extend(val_labels.tolist())
                predicted_labels.extend(predict_y.tolist())
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "Valid Epoch [{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        val_accuracy_list.append(val_accurate)
        train_loss_list.append(running_loss / train_steps)

        # 计算召回率、精确度和F1分数
        recall = recall_score(true_labels, predicted_labels, average='macro')
        precision = precision_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')

        val_recall_list.append(recall)
        val_precision_list.append(precision)
        val_f1_list.append(f1)

        print(
            'Epoch [{}/{}], Train Loss: {:.3f}, FPS: {:.2f}, Validation Accuracy: {:.2f}%, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(
                epoch + 1, epochs, running_loss / train_steps, fps, val_accurate * 100, precision, recall, f1))
        
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished training.')
    return train_loss_list, val_accuracy_list, val_recall_list, val_precision_list, val_f1_list

if __name__ == '__main__':
    train_loss_list, val_accuracy_list, val_recall_list, val_precision_list, val_f1_list = main()

    # 绘制训练损失曲线
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    # 绘制验证集准确率曲线
    plt.plot(val_accuracy_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.show()

    # 绘制召回率曲线
    plt.plot(val_recall_list, label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    plt.legend()
    plt.show()

    # 绘制精确度曲线
    plt.plot(val_precision_list, label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision Curve')
    plt.legend()
    plt.show()

    # 绘制F1分数曲线
    plt.plot(val_f1_list, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.show()
