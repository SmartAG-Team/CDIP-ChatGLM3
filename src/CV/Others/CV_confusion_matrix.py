import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import pandas as pd  # 用于保存CSV
import torch.utils.data as data
import sys
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
from model.CV_Model_Config.EfficientNet import efficientnet_b2 as create_model
from tqdm import tqdm  # 导入tqdm

class CustomImageDataset(data.Dataset):
    def __init__(self, data_dir, class_indict, transform=None):
        self.data_dir = data_dir
        self.class_indict = class_indict
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # 遍历目录，收集所有图像的路径和标签
        for label, class_name in class_indict.items():
            class_folder = os.path.join(data_dir, class_name)
            if os.path.isdir(class_folder):
                for image_name in os.listdir(class_folder):
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif')):
                        self.image_paths.append(os.path.join(class_folder, image_name))
                        self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def save_full_cm_to_csv(cm, class_names, filename):
    """将完整的混淆矩阵保存到 CSV 文件。"""
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(filename, index=True)
    print(f"Saved full confusion matrix to {filename}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图像预处理
    data_transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载标签映射文件
    json_path = './model/CV_Model_Config/class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 加载模型
    model = create_model(num_classes=61).to(device)
    model_weight_path = "./model/CV_Models/trained_model/efficientnetb2.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 创建数据集和数据加载器
    test_dir = "./data/CV/CV-dataset/test"
    dataset = CustomImageDataset(test_dir, class_indict, transform=data_transform)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_targets, all_predictions = [], []

    # 遍历测试集，使用批量推理
    for images, labels in tqdm(dataloader, desc="Processing Batches", unit="batch"):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, dim=1)

        all_targets.extend(labels.cpu().numpy())
        all_predictions.extend(predicted_classes.cpu().numpy())

    # 计算完整混淆矩阵
    class_names = [class_indict[str(i)] for i in range(len(class_indict))]
    cm = confusion_matrix(all_targets, all_predictions, labels=range(len(class_indict)))

    # 保存混淆矩阵到 CSV
    save_full_cm_to_csv(cm, class_names, "./data/CV/CV_confusion_matrix.csv")


if __name__ == '__main__':
    main()
