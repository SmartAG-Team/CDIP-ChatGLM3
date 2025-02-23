import os
import base64
import json
import requests
import difflib
import argparse

# ---------------------------
# 配置部分
# ---------------------------
# 父文件夹路径，每个子文件夹名称对应一个病害类别，且名称必须在 allowed_options 中
parser = argparse.ArgumentParser(description="Argument parser example")
parser.add_argument('--model', type=str, default='minicpm-v:8b', help='model name')
parser.add_argument('--delay', type=int, default=0, help='api call intervals')
args = parser.parse_args()

parent_folder = "./data/CV/Image Understanding Test Dataset"

# 允许的答案列表（注意：子文件夹名称应与其中某一项完全一致）
allowed_options = [
    "Apple alternaria boltch", "Apple black rot", "Apple brown spot", "Apple grey spot", "Apple healthy",
    "Apple mosaic virus", "Apple powdery mildew", "Apple rust", "Cherry healthy", "Cherry powdery mildew",
    "Citrus healthy", "Citrus huanglongbing", "Corn gray leaf spot", "Corn healthy", "Corn northern leaf blight",
    "Corn rust", "Grape black measles", "Grape black rot", "Grape downy mildew", "Grape heathy",
    "Grape leaf blight", "Grape mosaic virus", "Grape powdery mildew", "Grape yellows", "Peach bacterial spot",
    "Peach healthy", "Pepper bell bacterial spot", "Pepper healthy", "Pepper scab", "Potato healthy",
    "Potato late blight", "Rice bacterial blight", "Rice brown spot", "Rice healthy", "Rice hispa",
    "Rice leaf blast", "Rice tungro", "Soybean angular leaf spot", "Soybean bacterial blight",
    "Soybean cercospora leaf blight", "Soybean downy mildew", "Soybean frogeye", "Soybean healthy",
    "Soybean potassium deficiency", "Soybean rust", "Soybean target spot", "Strawberry healthy",
    "Strawberry leaf scorch", "Tomato bacterial spot", "Tomato early blight", "Tomato healthy",
    "Tomato late blight", "Tomato leaf mold", "Tomato mosaic virus", "Tomato septoria leaf spot",
    "Tomato yellow leaf curl virus", "Wheat brown rust", "Wheat healthy", "Wheat loose smut",
    "Wheat septoria", "Wheat yellow rust"
]
allowed_options_str = ", ".join(allowed_options)

input_text = (
    "What disease is in the picture? "
    "Please select ONE and ONLY ONE option from the following list and output EXACTLY that option without any extra words: "
    + allowed_options_str
)

# API 服务地址（确保服务已启动）
api_url = "http://localhost:11434/api/generate"

# 结果文件路径
output_file = "./data/CV/CV_Metric/VLM-Test-Metric/minicpm-v-8b.txt"                              

# ---------------------------
# 记录变量
# ---------------------------
total_images = 0
correct_total = 0
all_results = []

# 初始化每个类别的统计信息
metrics = {category: {"TP": 0, "FN": 0, "FP": 0, "Total": 0} for category in allowed_options}

# ---------------------------
# 处理每个子文件夹
# ---------------------------
subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
num_folders = len(subfolders)

for idx, subfolder in enumerate(subfolders, start=1):
    subfolder_path = os.path.join(parent_folder, subfolder)
    ground_truth = subfolder.strip()

    if ground_truth not in allowed_options:
        print(f"Warning: Subfolder name '{ground_truth}' not in allowed_options. Skipping.")
        continue

    print(f"\n[{idx}/{num_folders}] Processing folder: {ground_truth}")

    for image_file in os.listdir(subfolder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            total_images += 1
            metrics[ground_truth]["Total"] += 1
            image_path = os.path.join(subfolder_path, image_file)
            print(f"  Processing image: {image_file}")

            try:
                with open(image_path, "rb") as img_file:
                    image_data = img_file.read()
                image_b64 = base64.b64encode(image_data).decode("utf-8")
            except Exception as e:
                print(f"    Failed to process {image_file}: {e}")
                all_results.append({"folder": ground_truth, "image": image_file, "predicted": "Image read error", "correct": False})


            payload = {
                "model": "minicpm-v:8b",
                "prompt": input_text,
                "images": [image_b64]
            }

            try:
                response = requests.post(api_url, json=payload, timeout=60, stream=True)
                response.raise_for_status()

                accumulated_response = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            data = json.loads(line)
                            accumulated_response += data.get("response", "")
                        except Exception as parse_error:
                            print(f"    Error parsing JSON line: {line}\n    Error: {parse_error}")

                output_text = accumulated_response.strip()
                print(f"    Raw output: {output_text}")

                if output_text not in allowed_options:
                    matches = difflib.get_close_matches(output_text, allowed_options, n=1, cutoff=0.6)
                    output_text = matches[0] if matches else "No valid response found."

            except requests.exceptions.RequestException as e:
                print(f"    Request error for {image_file}: {e}")
                output_text = f"Request error: {e}"

            is_correct = (output_text == ground_truth)
            if is_correct:
                correct_total += 1
                metrics[ground_truth]["TP"] += 1
            else:
                metrics[ground_truth]["FN"] += 1
                if output_text in allowed_options:
                    metrics[output_text]["FP"] += 1

            all_results.append({"folder": ground_truth, "image": image_file, "predicted": output_text, "correct": is_correct})
            print(f"    Final prediction: '{output_text}' | Correct: {is_correct}")

    # 每处理完一个文件夹就写入
    with open(output_file, "a", encoding="utf-8") as f:
        for result in all_results:
            f.write(f"{result['folder']}/{result['image']} --> Predicted: {result['predicted']} | Correct: {result['correct']}\n")
        all_results.clear()  # 清空缓存，避免重复写入

# ---------------------------
# 计算整体指标
# ---------------------------
with open(output_file, "a", encoding="utf-8") as f:
    f.write("\nPer-Class Metrics:\n")
    total_TP = total_FP = total_FN = 0

    for category, stats in metrics.items():
        TP, FP, FN, Total = stats["TP"], stats["FP"], stats["FN"], stats["Total"]
        total_TP += TP
        total_FP += FP
        total_FN += FN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = TP / Total if Total > 0 else 0

        f.write(f"{category} | ACC: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}\n")

    overall_accuracy = correct_total / total_images if total_images > 0 else 0
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    f.write(f"\nOverall Metrics: ACC: {overall_accuracy:.2f}, Precision: {overall_precision:.2f}, Recall: {overall_recall:.2f}, F1: {overall_f1:.2f}\n")

print("\nBatch processing completed!")
