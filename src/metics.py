import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载训练数据集
with open("./data/fine-tuning_dataset/test/grape_test_data.json", "r", encoding="utf-8") as json_file:
    data_source = json.load(json_file)
print(data_source)

# 设置输出文件路径
output_file_path = './data/response/chatglm_grape_apple/chatglm_grape.json'

# 如果目录不存在，创建目录
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# 如果文件不存在，创建一个空的 JSON 文件
if not os.path.exists(output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump([], json_file, ensure_ascii=False, indent=4)

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(
    r"D:\student\lzy\llM\LLaMA-Factory-main\chatglm3_6b", 
    trust_remote_code=True
)
gpt_model = AutoModelForCausalLM.from_pretrained(
    r"D:\student\lzy\llM\LLaMA-Factory-main\chatglm3_6b", 
    trust_remote_code=True, 
    device='cuda'
)
gpt_model = gpt_model.to('cuda').eval()

# 读取原有的数据（如果有）
with open(output_file_path, 'r', encoding='utf-8') as json_file:
    existing_data = json.load(json_file)

# 生成新数据
new_data = []
for item in data_source:
    instruction = item["instruction"]
    # 使用模型生成回复
    response, _ = gpt_model.chat(tokenizer, instruction, history=[])
    
    entry = {
        "instruction": instruction,
        "input": "",
        "output": response
    }
    print(entry)
    new_data.append(entry)

# 合并新生成的数据和现有数据
combined_data = existing_data + new_data

# 将合并后的数据写入文件
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(combined_data, json_file, ensure_ascii=False, indent=4)

print("生成的回复已保存")
