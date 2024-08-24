import os
import json
import torch
from PIL import Image
from torchvision import transforms
import time
from sklearn.metrics import average_precision_score, recall_score, f1_score, precision_score
import numpy as np
from thop import profile

from vit_model import vit_base_patch32_224_in21k as create_model


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

        # Only calculate scores if there is at least one positive example
        if sum(class_targets) > 0:
            ap = average_precision_score(class_targets, class_predictions)
            recall = recall_score(class_targets, class_predictions, zero_division=0)
            f1 = f1_score(class_targets, class_predictions, zero_division=0)
            precision = precision_score(class_targets, class_predictions, zero_division=0)
        else:
            ap = 0
            recall = 0
            f1 = 0
            precision = 0

        ap_scores.append(ap)
        recall_scores.append(recall)
        f1_scores.append(f1)
        precision_scores.append(precision)

    mAP = np.mean(ap_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)

    return mAP, avg_recall, avg_f1, avg_precision


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    folder_path = "..\\test"
    assert os.path.exists(folder_path), "folder_path: '{}' does not exist.".format(folder_path)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    class_indict = read_class_indices(json_path)

    class_names = list(class_indict.values())

    model = create_model(num_classes=52, has_logits=False)
    model_weight_path = "..\\vit16-224ink.pth"
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

    total_correct = 0
    total_images = 0

    start_time = time.time()  # Record start time for calculating FPS

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        print(f"\nPredictions for images in {subfolder_path}")

        correct_count = 0  # Track the number of correct predictions

        for image_name in os.listdir(subfolder_path):
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

            print_res = "Folder: {}  Image: {}".format(subfolder, image_path)
            print_res += "\nOriginal Class {}\tPredicted Class: {}".format(subfolder, predicted_class)

            for class_idx, class_name in enumerate(class_names):
                print_res += "\n{}\t\t{:.3}".format(class_name, predict[class_idx].item())

            # print(print_res)

            if predicted_class == subfolder:
                correct_count += 1

        accuracy = correct_count / len(os.listdir(subfolder_path))
        print("Accuracy for folder {}: {:.2%}".format(subfolder, accuracy))

        total_correct += correct_count
        total_images += len(os.listdir(subfolder_path))

    end_time = time.time()  # Record end time for calculating FPS
    elapsed_time = end_time - start_time

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    print("Overall Accuracy: {:.2%}".format(overall_accuracy))

    # Calculate FLOPs
    input_data = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input_data,))
    print(f"FLOPs: {flops / 1e9:.3f} G")

    # Compute metrics
    mAP, avg_recall, avg_f1, avg_precision = compute_metrics(predictions, targets, class_names)
    print("mAP: {:.3}".format(mAP))
    print("Recall: {:.3}".format(avg_recall))
    print("F1: {:.3}".format(avg_f1))
    print("Precision: {:.3}".format(avg_precision))

    fps = total_images / elapsed_time if elapsed_time > 0 else 0
    print("FPS: {:.2f}".format(fps))


if __name__ == '__main__':
    main()
