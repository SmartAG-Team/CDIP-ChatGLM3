import os
import json
import re
import pandas as pd
from openai import OpenAI
import argparse
import time
import csv
from dotenv import load_dotenv
load_dotenv()

# 初始化 Argument Parser
parser = argparse.ArgumentParser(description="Argument parser example")
parser.add_argument('--model', type=str, default='llama3.1-405b-instruct', help='model name')
parser.add_argument('--delay', type=int, default=5, help='api call intervals')
parser.add_argument('--api_key', type=str, default=os.getenv('Llama_API_KEY'), help='api key') 
args = parser.parse_args()

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=args.api_key,  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
)

csv_file_path = './data/LLM/LLM_dataset/Specialized-Abilities-Test-Dataset/disease_prevention_question.csv'  # 替换为 CSV 文件路径
output_file_path = './data/LLM/LLM_dataset/Specialized-Abilities-Test-Dataset/Specialized-Abilities-Test-Dataset.csv'  # 输出CSV路径

def generate_output_from_csv(csv_file_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    # 如果输出文件不存在，创建文件并写入表头
    if not os.path.exists(output_file_path):
        with open(output_file_path, mode="w", newline='', encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(['disease','question', 'question1', 'question2', 'question3', 'question4', 'answer'])
    df2 = pd.read_csv(output_file_path)

    # 获取已存在的 'question' 列
    questions_in_df2 = df2['question'].to_list()

    # 实时写入输出结果到 CSV 文件
    for _, row in df.iterrows():
        question = row['question']
        print(question)
        if question in questions_in_df2:
            print(f"问题 '{question}' 已经存在，跳过。")
            continue
        try:
            answer = row['answer']
            disease = row['disease']
            answer = answer.replace("\n", "\\n")  # 将换行符替换为 \n

            completion = client.chat.completions.create(
                model=args.model,
                messages=[{
                    'role': 'user',
                    'content': f'请将这句话改写为五个不同的句子，要求所有句子为命令句或问题句，并且每个句子在结构、语态或词语使用上必须显著不同于原句。不能重复原句，否则将会被扣分，且不要使用感叹句。请将结果放入数组中，格式为：["","","","",""]。原句为: {question}。'
                }],
                stream=False
            )

            response = completion.choices[0].message.model_dump()
            content = response['content']

            # 检查并清理 API 响应中的多余字符
            if content.startswith("[") and content.endswith("]"):
                parsed_content = json.loads(content)
            else:
                print(f"API 响应格式不符合 JSON 数组格式: {content}")
                continue
            
            # 检查并移除与原句相同的句子
            unique_questions = [q for q in parsed_content if q != question]
            
            # 如果去重后少于4条，则重新生成
            while len(unique_questions) < 4:
                additional_completion = client.chat.completions.create(
                    model=args.model,
                    messages=[{
                        'role': 'user',
                        'content': f'请再生成一些不同的句子，以确保句子总数达到四个，且与原句不同。'
                    }],
                    stream=False
                )
                additional_response = additional_completion.choices[0].message.model_dump()
                additional_content = additional_response['content']
                
                if additional_content.startswith("[") and additional_content.endswith("]"):
                    additional_parsed_content = json.loads(additional_content)
                    unique_questions += [q for q in additional_parsed_content if q != question]
                    unique_questions = list(set(unique_questions))[:4]

            print(unique_questions[:4])
            with open(output_file_path, mode="a", newline='', encoding="utf-8") as output_file:
                writer = csv.writer(output_file)
                writer.writerow([disease] + [question] + unique_questions[:4] + [answer])  # 确保只写入四条句子

            time.sleep(args.delay)

        except Exception as e:
            print(f"An error occurred: {e}")
            break


if __name__ == "__main__":
    generate_output_from_csv(csv_file_path)
