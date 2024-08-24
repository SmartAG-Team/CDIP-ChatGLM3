import os
import json
import torch
from PIL import Image
from torchvision import transforms
import time
from sklearn.metrics import average_precision_score, recall_score, f1_score, accuracy_score, precision_score
import numpy as np
from thop import profile

from vit_model import vit_base_patch16_224 as create_model

def read_class_indices(json_path):
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    return class_indict

def compute_metrics(predictions, targets, class_names):
    num_classes = len(class_names)
    ap_scores = []
    recall_scores = []
    f1_scores = []
    precision_scores = []

    for class_idx in range(num_classes):
        class_name = class_names[class_idx]
        class_predictions = [1 if p == class_name else 0 for p in predictions]
        class_targets = [1 if t == class_name else 0 for t in targets]

        ap = average_precision_score(class_targets, class_predictions)
        recall = recall_score(class_targets, class_predictions, zero_division=0)
        f1 = f1_score(class_targets, class_predictions, zero_division=0)
        precision = precision_score(class_targets, class_predictions, zero_division=0)

        ap_scores.append(ap)
        recall_scores.append(recall)
        f1_scores.append(f1)
        precision_scores.append(precision)

    mAP = np.mean(ap_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)
    accuracy = accuracy_score(targets, predictions)

    return mAP, avg_recall, avg_f1, avg_precision, accuracy

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    folder_path = "..\\test"
    assert os.path.exists(folder_path), "folder_path: '{}' does not exist.".format(folder_path)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    class_indict = read_class_indices(json_path)

    class_names = list(class_indict.values())

    model = create_model(num_classes=52)
    model_weight_path = "..\\vit-16_224.pth"
    assert os.path.exists(model_weight_path), "model_weight_path: '{}' does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    model.eval()

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    predictions = []
    targets = []
    correct_predictions = 0
    total_images = 0

    start_time = time.time()

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        print(f"\nPredictions for images in {subfolder_path}")

        for image_name in os.listdir(subfolder_path):
            total_images += 1
            image_path = os.path.join(subfolder_path, image_name)
            assert os.path.exists(image_path), "file: '{}' does not exist.".format(image_path)
            img = Image.open(image_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).item()

            predicted_class = class_indict[str(predict_cla)]
            predictions.append(predicted_class)
            targets.append(subfolder)

            if predicted_class == subfolder:
                correct_predictions += 1

            print_res = "Folder: {}  Image: {}  Predicted Class: {}  Probability: {:.3}".format(
                subfolder, image_path, predicted_class, predict[predict_cla].item())
            print(print_res)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate FLOPs
    input_data = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input_data,))
    print(f"FLOPs: {flops / 1e9} G")

    # Compute metrics
    mAP, avg_recall, avg_f1, avg_precision, accuracy = compute_metrics(predictions, targets, class_names)
    print("mAP: {:.3}".format(mAP))
    print("Recall: {:.3}".format(avg_recall))
    print("F1: {:.3}".format(avg_f1))
    print("Precision: {:.3}".format(avg_precision))
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    fps = total_images / elapsed_time
    print("FPS: {:.2f}".format(fps))

if __name__ == '__main__':
    main()
