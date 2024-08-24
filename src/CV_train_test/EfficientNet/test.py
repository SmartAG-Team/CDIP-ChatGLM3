import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import efficientnet_b0 as create_model
from sklearn.metrics import recall_score, f1_score, average_precision_score, precision_score
import numpy as np
import csv

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # img_size = {"B0": 224, "B1": 240, "B2": 260, "B3": 300, "B4": 380, "B5": 456, "B6": 528, "B7": 600}
    # num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(300),
         transforms.CenterCrop(300),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    correct_predictions_total = 0
    total_images_total = 0

    all_targets = []
    all_predictions = []

    # Model initialization outside of loop
    model = create_model(num_classes=52).to(device)
    model_weight_path = "..\\efficientnet_b0.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # Set batch size
    batch_size = 32

    # Loop through each subfolder in the test directory
    test_dir = "..\\test"
    for folder in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Predicting images in folder: {folder}")
            folder_correct_predictions = 0
            folder_total_images = 0

            image_batch = []
            target_batch = []

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if image_path.lower().endswith((".jpg", ".jfif", ".jpeg", ".png")):
                    try:
                        img = Image.open(image_path)
                        
                        # Convert image to RGB if it has an alpha channel (RGBA)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        img = data_transform(img)
                        image_batch.append(img)
                        target_batch.append(folder)

                        # Process batch if it reaches the batch size
                        if len(image_batch) == batch_size:
                            image_batch = torch.stack(image_batch).to(device)
                            with torch.no_grad():
                                output = model(image_batch)
                                probs = torch.softmax(output, dim=1).cpu().numpy()
                                predicted_classes = torch.argmax(output, dim=1).cpu().numpy()

                            for i in range(batch_size):
                                actual_label = target_batch[i]
                                predicted_label = class_indict[str(predicted_classes[i])]

                                all_targets.append(int(list(class_indict.keys())[list(class_indict.values()).index(actual_label)]))
                                all_predictions.append(int(list(class_indict.keys())[list(class_indict.values()).index(predicted_label)]))

                                if actual_label == predicted_label:
                                    folder_correct_predictions += 1
                                    correct_predictions_total += 1

                            folder_total_images += len(image_batch)
                            image_batch = []
                            target_batch = []

                    except Exception as e:
                        print(f"Skipping image {image_name} due to error: {e}")

            # Process remaining images in batch
            if len(image_batch) > 0:
                image_batch = torch.stack(image_batch).to(device)
                with torch.no_grad():
                    output = model(image_batch)
                    probs = torch.softmax(output, dim=1).cpu().numpy()
                    predicted_classes = torch.argmax(output, dim=1).cpu().numpy()

                for i in range(len(image_batch)):
                    actual_label = target_batch[i]
                    predicted_label = class_indict[str(predicted_classes[i])]

                    all_targets.append(int(list(class_indict.keys())[list(class_indict.values()).index(actual_label)]))
                    all_predictions.append(int(list(class_indict.keys())[list(class_indict.values()).index(predicted_label)]))

                    if actual_label == predicted_label:
                        folder_correct_predictions += 1
                        correct_predictions_total += 1

                folder_total_images += len(image_batch)

            # Calculate accuracy for this folder and print
            folder_accuracy = folder_correct_predictions / folder_total_images if folder_total_images != 0 else 0
            print(f"Folder: {folder}, Accuracy: {folder_accuracy:.2%}, Total images: {folder_total_images}, Correct predictions: {folder_correct_predictions}")
 
            total_images_total += folder_total_images

    # Calculate overall accuracy and print
    overall_accuracy = correct_predictions_total / total_images_total if total_images_total != 0 else 0
    print("Overall Accuracy: {:.2%}".format(overall_accuracy))
    print("Total correct predictions: {}, Total images: {}".format(correct_predictions_total, total_images_total))

    # Calculate Precision, Recall, and F1-score
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    print(f"Overall Precision: {precision:.2%}")
    print(f"Overall Recall: {recall:.2%}")
    print(f"Overall F1-Score: {f1:.2%}")

    # Save the results to a CSV file
    output_csv = 'results.csv'
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'Model': 'EfficientNet-B0',
            'Accuracy': overall_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

if __name__ == '__main__':
    main()
