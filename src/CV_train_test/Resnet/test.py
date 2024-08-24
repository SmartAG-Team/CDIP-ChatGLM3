import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import resnet101
import csv
import time
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np

def read_class_indices(json_path):
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    return class_indict

def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    unique_classes = np.unique(y_true)
    ap_list = []
    for cls in unique_classes:
        y_true_binary = [1 if label == cls else 0 for label in y_true]
        y_pred_binary = [1 if pred == cls else 0 for pred in y_pred]
        ap = average_precision_score(y_true_binary, y_pred_binary)
        ap_list.append(ap)
    
    mAP = np.mean(ap_list)
    return precision, recall, f1, mAP

def predict_folder(folder_path, model, class_indict, device, output_folder, model_name):
    correct_predictions = 0
    total_predictions = 0
    total_images = 0
    total_time = 0

    all_true_labels = []
    all_predicted_labels = []

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)
            assert os.path.exists(image_path), "file: '{}' does not exist.".format(image_path)
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            start_time = time.time()
            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time

            predicted_class = class_indict[str(predict_cla)]
            all_true_labels.append(subfolder)
            all_predicted_labels.append(predicted_class)

            if predicted_class == subfolder:
                correct_predictions += 1
            total_predictions += 1
            total_images += 1

    accuracy = correct_predictions / total_predictions
    avg_fps = total_images / total_time
    precision, recall, f1, mAP = compute_metrics(all_true_labels, all_predicted_labels)

    # Print metrics
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.3}")
    print(f"Precision: {precision:.3}")
    print(f"Recall: {recall:.3}")
    print(f"F1: {f1:.3}")


    # Save results to CSV file
    csv_file_path = os.path.join(output_folder, 'model_metrics.csv')
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['model_name', 'accuracy', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_path = ".\\test"
    json_path = './class_indices.json'
    class_indict = read_class_indices(json_path)

    model = resnet101(num_classes=52).to(device)
    model_weight_path = "resNet101.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    output_folder = './predictions'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    predict_folder(folder_path, model, class_indict, device, output_folder, model_name="ResNet50")
