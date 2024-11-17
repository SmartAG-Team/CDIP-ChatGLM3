from transformers import GPT2Tokenizer
import os
import json
import csv
import hashlib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
rouge = Rouge()
import csv
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
directory = './data/LLM/LLM_Model_Response'

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./model/LLM_models/ChatGLM3_6B", 
    trust_remote_code=True
)

# 计算token数量
def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# 替换现有的len(output_text)逻辑为token数量逻辑
def process_folder(directory):
    model_error_rates = {}
    for root, dirs, files in os.walk(directory):
        model_name = os.path.basename(root)  # 初始化 model_name
        error_count = 0
        total_count = 0
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as json_file:
                    data_source = json.load(json_file)
                    total_count += len(data_source)
                    for i in data_source:
                        output_text = i.get('output', '')
                        if count_tokens(output_text) > 5000:  # 使用token数量判断
                            error_count += 1
        if total_count > 0:  # 确保不会除以零
            error_rate = (error_count / total_count) * 100
            model_error_rates[model_name] = error_rate
    return model_error_rates

error_rates = process_folder(directory)

csv_filename = './data/LLM/LLM_Metric/Error-rate-Metric/Error-rate-Metric-token.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Model', 'Error Rate (%)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for model_name, error_rate in error_rates.items():
        writer.writerow({
            'Model': model_name,
            'Error Rate (%)': f"{error_rate:.2f}"
        })

print(f"Error rate information has been saved to {csv_filename}")

