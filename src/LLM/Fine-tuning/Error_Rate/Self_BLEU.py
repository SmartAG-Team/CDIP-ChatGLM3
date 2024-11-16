import os
import json
import csv
import nltk
import concurrent.futures
from tqdm import tqdm  # 引入tqdm库用于显示进度条
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# 确保下载 punkt
nltk.download('punkt')

def compute_self_bleu(outputs):
    """
    计算Self-BLEU分数，outputs是模型输出的列表。
    """
    bleu_scores = []
    for i in tqdm(range(len(outputs)), desc="计算Self-BLEU进度"):  # 添加进度条
        refs = outputs[:i] + outputs[i+1:]  # 其他输出作为参考
        refs = [word_tokenize(ref) for ref in refs]  # 参考数据进行分词
        candidate = word_tokenize(outputs[i])  # 计算BLEU分数的候选输出
        if len(candidate) > 0 and len(refs) > 0:
            score = sentence_bleu(refs, candidate, weights=(0.25, 0.25, 0.25, 0.25))
            bleu_scores.append(score)
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

def process_json_files_in_folder_parallel(folder_path):
    """
    并行处理文件夹中的所有JSON文件，提取输出并计算平均Self-BLEU分数。
    """
    all_outputs = []
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    def process_json(file_name):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return [item.get('output', '').strip() for item in data if item.get('output', '').strip()]

    # 使用进度条显示文件处理进度
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_json, json_files), total=len(json_files), desc="处理JSON文件"))

    for result in results:
        all_outputs.extend(result)

    if all_outputs:
        avg_self_bleu = compute_self_bleu(all_outputs)
        return avg_self_bleu
    else:
        return 0

def read_existing_models(csv_file):
    """
    从CSV文件中读取已经处理过的模型名称。
    """
    if not os.path.exists(csv_file):
        return set()  # 如果文件不存在，返回一个空集合
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)  # 跳过表头
        return {row[0] for row in reader}  # 返回已处理模型的集合

def process_main_folder(main_folder):
    """
    遍历主文件夹的所有子文件夹，计算每个模型文件夹的平均Self-BLEU分数，并保存为CSV。
    """
    csv_file = os.path.join(main_folder, 'Avg_self_bleu_scores.csv')

    # 获取所有子文件夹
    folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

    # 读取已处理的模型
    processed_models = read_existing_models(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:  # 使用附加模式打开文件
        writer = csv.writer(file)
        if not processed_models:
            writer.writerow(['Model', 'Self-BLEU'])  # 如果CSV文件是新建的，写入表头

        # 使用进度条显示文件夹处理进度
        for folder_name in tqdm(folders, desc="处理文件夹"):
            if folder_name in processed_models:
                print(f"模型 {folder_name} 已处理，跳过。")
                continue  # 跳过已处理的模型

            folder_path = os.path.join(main_folder, folder_name)
            avg_self_bleu = process_json_files_in_folder_parallel(folder_path)
            writer.writerow([folder_name, avg_self_bleu])

    print(f"结果已保存到: {csv_file}")

if __name__ == "__main__":
    # 设置主文件夹路径
    main_folder_path = './data/LLM/LLM_Model_Response'  # 替换为你的主文件夹路径
    process_main_folder(main_folder_path)
