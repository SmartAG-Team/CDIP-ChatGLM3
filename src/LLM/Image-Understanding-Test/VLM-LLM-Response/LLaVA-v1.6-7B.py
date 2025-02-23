import os
import base64
import difflib
import pandas as pd
import argparse
import time
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import json

parser = argparse.ArgumentParser(description="Argument parser example")
parser.add_argument('--model', type=str, default='llava:7b', help='model name')
parser.add_argument('--delay', type=int, default=0, help='api call intervals')
args = parser.parse_args()

# 输出CSV文件路径（将生成回复后的 CSV 保存到该文件）
output_file_path = './data/LLM/LLM_Metric/Specialized-Abilities-VLM-Test-Metric/VLM_Response/Llava-1.6-7b.csv'
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# 读取原始 CSV 文件（假设文件存在）
file_path = './data/LLM/LLM_dataset/Specialized-Abilities-Test-Dataset/Specialized-Abilities-Test-Dataset.csv'
if not os.path.exists(output_file_path):
    df = pd.read_csv(file_path)
    df.to_csv(output_file_path, index=False, encoding='utf-8')\
# ---------------------------
# ---------------------------
api_url = "http://localhost:11434/api/generate"

# ---------------------------
# 处理 CSV 数据前先加载 CSV 文件
df = pd.read_csv(output_file_path)

# 检查是否存在 answer, answer1, answer2, answer3, answer4 列
for col in ['answer', 'answer1', 'answer2', 'answer3', 'answer4']:
    if col not in df.columns:
        df[col] = ''

df.to_csv(output_file_path, index=False, encoding='utf-8')

# ---------------------------
# 遍历 CSV 文件中的每一行，调用 API 生成回复
# ---------------------------
for index, row in df.iterrows():
    updated = False

    # 构造一个函数，统一调用接口（复用逻辑）
    def call_api(instruction):
        # 构造消息内容（先传文本提示，再传图片，如果有图片，请自行修改此处）
        payload = {
            "model": args.model,
            "prompt": instruction,
        }
        try:
            response = requests.post(api_url, json=payload, timeout=60, stream=True)
            response.raise_for_status()

            accumulated_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    data = json.loads(line)
                    accumulated_response += data.get("response", "")
            # 根据官方格式提取回复内容
            output_text = accumulated_response.strip()
        except Exception as e:
            print(f"    Request error: {e}")
            output_text = f"Request error: {e}"
        return output_text

    # 处理 question 列
    if pd.notna(row['question']) and pd.isna(row['answer']):
        instruction = row['question']
        response = call_api(instruction)
        response = response.replace("\n", "\\n")
        print(f"问题: {instruction}")
        print(f"答案: {response}")
        df.at[index, 'answer'] = response
        updated = True
        time.sleep(args.delay)

    # 处理 question1 列
    if pd.notna(row['question1']) and pd.isna(row['answer1']):
        instruction = row['question1']
        response = call_api(instruction)
        response = response.replace("\n", "\\n")
        print(f"问题: {instruction}")
        print(f"答案: {response}")
        df.at[index, 'answer1'] = response
        updated = True
        time.sleep(args.delay)

    # 处理 question2 列
    if pd.notna(row['question2']) and pd.isna(row['answer2']):
        instruction = row['question2']
        response = call_api(instruction)
        response = response.replace("\n", "\\n")
        print(f"问题: {instruction}")
        print(f"答案: {response}")
        df.at[index, 'answer2'] = response
        updated = True
        time.sleep(args.delay)

    # 处理 question3 列
    if pd.notna(row['question3']) and pd.isna(row['answer3']):
        instruction = row['question3']
        response = call_api(instruction)
        response = response.replace("\n", "\\n")
        print(f"问题: {instruction}")
        print(f"答案: {response}")
        df.at[index, 'answer3'] = response
        updated = True
        time.sleep(args.delay)

    # 处理 question4 列
    if pd.notna(row['question4']) and pd.isna(row['answer4']):
        instruction = row['question4']
        response = call_api(instruction)
        response = response.replace("\n", "\\n")
        print(f"问题: {instruction}")
        print(f"答案: {response}")
        df.at[index, 'answer4'] = response
        updated = True

    if updated:
        df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"已保存第 {index+1} 行的更新到 {output_file_path}")

