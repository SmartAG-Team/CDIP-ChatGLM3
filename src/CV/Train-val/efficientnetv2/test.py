import os
import json
import torch
import time
from PIL import Image
from torchvision import transforms
import sys
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
from model.CV_Model_Config.efficientnetv2 import efficientnetv2_s as create_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import csv
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform, class_indict):
        self.data_dir = data_dir
        self.transform = transform
        self.class_indict = class_indict
        self.image_paths = []
        self.labels = []

        # 准备图像路径和对应的标签
        label_to_index = {v: int(k) for k, v in class_indict.items()}
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                if folder not in label_to_index:
                    print(f"Skipping folder {folder} as it is not in class_indict.")
                    continue
                true_label = label_to_index[folder]
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    if image_path.lower().endswith((".jpg", ".jpeg", ".png", ".JPG", ".PNG")):
                        self.image_paths.append(image_path)
                        self.labels.append(true_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)
        return img, label

def calculate_params(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # 单位：百万(M)

def measure_latency(model, img, device, runs=50):
    """测量模型推理的平均延迟和 FPS"""
    img = img.to(device)
    model.eval()
    times = []

    with torch.no_grad():
        for _ in range(runs):
            start_time = time.time()
            _ = model(img)
            times.append(time.time() - start_time)

    avg_latency = (np.mean(times) * 1000)  # 毫秒(ms)
    fps = 1 / np.mean(times)
    return fps, avg_latency

def save_results_to_csv(file_path, results):
    """将结果保存到 CSV 文件"""
    headers = [
        'Model', 'Accuracy', 'Precision', 'Recall', 
        'F1-Score', 'Params (M)', 'FPS', 'Latency (ms)'
    ]

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(results)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"s": [300, 384], "m": [384, 480], "l": [384, 480]}
    num_model = "s"

    data_transform = transforms.Compose([
        transforms.Resize(img_size[num_model][1]),
        transforms.CenterCrop(img_size[num_model][1]),
        transforms.ToTensor()
    ])

    json_path = './model/CV_Model_Config/class_indices.json'
    assert os.path.exists(json_path), f"File '{json_path}' does not exist."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = create_model(num_classes=61).to(device)
    model_weight_path = "./model/CV_Models/trained_model_1/efficientnetv2s.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 创建数据集和数据加载器
    test_dir = "./data/CV/CV_dataset/test"
    dataset = CustomDataset(test_dir, data_transform, class_indict)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    total_preds, total_labels = [], []
    correct_predictions_total = 0
    total_images_total = 0

    # 批量推理
# 批量推理
    for images, labels in tqdm(data_loader, desc="Testing Progress", unit="batch"):
     images, labels = images.to(device), labels.to(device)

     with torch.no_grad():
        outputs = model(images)
        predicts = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

     total_preds.extend(predicts.cpu().numpy())
     total_labels.extend(labels.cpu().numpy())

     correct_predictions_total += (predicts == labels).sum().item()
     total_images_total += len(labels)


    # 计算性能指标
    overall_accuracy = correct_predictions_total / total_images_total if total_images_total != 0 else 0
    precision, recall, f1, _ = precision_recall_fscore_support(total_labels, total_preds, average='weighted')

    # 测量模型参数和延迟
    params = calculate_params(model)
    sample_img = torch.randn(1, 3, img_size[num_model][1], img_size[num_model][1]).to(device)
    fps, avg_latency = measure_latency(model, sample_img, device)

    print(f"Model: EfficientNetV2-S")
    print(f"Accuracy: {overall_accuracy:.2%}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"Params (M): {params:.2f}")
    print(f"FPS: {fps:.2f}")
    print(f"Latency (ms): {avg_latency:.2f}")

    results = [
        'EfficientNetV2', overall_accuracy, precision, recall, 
        f1, params, fps, avg_latency
    ]
    save_results_to_csv('./data/CV/CV_Metric/CV_results.csv', results)

if __name__ == '__main__':
    main()

