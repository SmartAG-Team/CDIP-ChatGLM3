import os
import json
import torch
import time
from PIL import Image
from torchvision import transforms
import sys
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)
from model.CV_Model_Config.mobilenetv3 import mobilenet_v3_small as create_model
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # 用于保存CSV
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 


# Dataset类，用于加载图像数据
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_indict = {}

        # Load class indices and paths
        json_path = './model/CV_Model_Config/class_indices.json'
        assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
        with open(json_path, "r") as f:
            self.class_indict = json.load(f)
        
        # Prepare image paths and labels
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    if image_path.lower().endswith((".jpg", ".jfif", ".jpeg", ".png")):
                        self.image_paths.append(image_path)
                        self.labels.append(folder)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # Convert label to class index
        label_index = int(list(self.class_indict.keys())[list(self.class_indict.values()).index(label)])
        return img, label_index

def measure_latency_and_fps(model, inputs, device, warmup=5, iterations=50):
    """Measure GPU latency and FPS."""
    for _ in range(warmup):  # Warmup to stabilize CUDA
        _ = model(inputs.to(device))

    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        _ = model(inputs.to(device))
        torch.cuda.synchronize()
    
    end = time.time()

    total_time = end - start
    avg_latency = total_time / iterations * 1000  # Convert to milliseconds
    fps = iterations / total_time
    return avg_latency, fps

def count_parameters(model):
    """Count the number of parameters in the model (in millions)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def save_full_cm_to_csv(cm, class_names, filename):
    """将完整的混淆矩阵保存到 CSV 文件。"""
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(filename, index=True)
    print(f"Saved full confusion matrix to {filename}")

def plot_confusion_matrix(cm, class_names):
    """Plot the confusion matrix."""
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data transformations
    data_transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize model
    model = create_model(num_classes=61).to(device)
    model_weight_path = "./model/CV_Models/trained_model_1/mobilenet.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # Measure model params and prepare dummy input for latency/FPS
    params = count_parameters(model)
    dummy_input = torch.randn(1, 3, 300, 300).to(device)
    avg_latency, fps = measure_latency_and_fps(model, dummy_input, device)

    print(f"Model Parameters: {params:.2f}M")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"FPS: {fps:.2f}")

    # Prepare dataset and dataloader
    test_dir = "./data/CV/CV_dataset/test"
    dataset = CustomDataset(test_dir, transform=data_transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    correct_predictions_total = 0
    total_images_total = 0
    all_targets = []
    all_predictions = []

    for images, labels in tqdm(data_loader, desc="Processing Batches"):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            output = model(images)
            predicted_classes = torch.argmax(output, dim=1)

        # Update targets and predictions
        all_targets.extend(labels.cpu().numpy())
        all_predictions.extend(predicted_classes.cpu().numpy())

        correct_predictions_total += (predicted_classes == labels).sum().item()
        total_images_total += len(labels)

    overall_accuracy = correct_predictions_total / total_images_total if total_images_total != 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    print(f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1-Score: {f1:.2%}")


    output_csv = './data/CV/CV_Metric/CV_results.csv'
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Params (M)', 'FPS', 'Latency (ms)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'Model': 'MobileNetV3-Small',
            'Accuracy': overall_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Params (M)': f"{params:.2f}",
            'FPS': fps,
            'Latency (ms)': avg_latency
        })

if __name__ == '__main__':
    main()
