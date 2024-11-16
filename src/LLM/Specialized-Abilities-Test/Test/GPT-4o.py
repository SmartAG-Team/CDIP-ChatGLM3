import os
import pandas as pd
import openai
import argparse
import time
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser(description="Argument parser example")
parser.add_argument('--api_key', type=str, default=os.getenv('OpenAI_API_KEY'), help='API key') 
parser.add_argument('--model', type=str, default='gpt-4o', help='Model name')
parser.add_argument('--delay', type=int, default=3, help='API call intervals')
parser.add_argument('--base_url',type=str,default=os.getenv('OpenAI_API_base_url'))
args = parser.parse_args()

def generate_responses_from_csv(output_file_path):
    openai.api_key = args.api_key
    openai.base_url = args.base_url
    df = pd.read_csv(output_file_path)

    for col in ['answer', 'answer1', 'answer2', 'answer3', 'answer4']:
            if col not in df.columns:
                df[col] = ''  
    df.to_csv(output_file_path, index=False, encoding='utf-8')

    # 遍历每一行，处理问题并生成LLM的回复
    for index, row in df.iterrows():
        updated = False
        # 处理 question 列
        if pd.notna(row['question']) and pd.isna(row['answer']):  
            instruction = row['question']

            completion = openai.ChatCompletion.create(
                model=args.model,
                messages=[{'role': 'user', 'content': instruction}],
                stream=False
            )
            response = completion.choices[0].message['content']
            response = response.replace("\n", "\\n")  
            print(f"问题: {instruction}")
            print(f"答案: {response}")
            df.at[index, 'answer'] = response
            updated = True
            time.sleep(args.delay)
        
        # 处理 question1 列
        if pd.notna(row['question1']) and pd.isna(row['answer1']):
            instruction = row['question1']

            completion = openai.ChatCompletion.create(
                model=args.model,
                messages=[{'role': 'user', 'content': instruction}],
                stream=False
            )
            response = completion.choices[0].message['content']
            response = response.replace("\n", "\\n")  
            print(f"问题: {instruction}")
            print(f"答案: {response}")
            df.at[index, 'answer1'] = response
            updated = True
            time.sleep(args.delay)

        # 处理 question2 列
        if pd.notna(row['question2']) and pd.isna(row['answer2']):
            instruction = row['question2']

            completion = openai.ChatCompletion.create(
                model=args.model,
                messages=[{'role': 'user', 'content': instruction}],
                stream=False
            )
            response = completion.choices[0].message['content']
            response = response.replace("\n", "\\n")  
            print(f"问题: {instruction}")
            print(f"答案: {response}")
            df.at[index, 'answer2'] = response
            updated = True
            time.sleep(args.delay)

        # 处理 question3 列
        if pd.notna(row['question3']) and pd.isna(row['answer3']):
            instruction = row['question3']

            completion = openai.ChatCompletion.create(
                model=args.model,
                messages=[{'role': 'user', 'content': instruction}],
                stream=False
            )
            response = completion.choices[0].message['content']
            response = response.replace("\n", "\\n")  
            print(f"问题: {instruction}")
            print(f"答案: {response}")
            df.at[index, 'answer3'] = response
            updated = True
            time.sleep(args.delay)

        # 处理 question4 列
        if pd.notna(row['question4']) and pd.isna(row['answer4']):
            instruction = row['question4']

            completion = openai.ChatCompletion.create(
                model=args.model,
                messages=[{'role': 'user', 'content': instruction}],
                stream=False
            )
            response = completion.choices[0].message['content']
            response = response.replace("\n", "\\n")  
            print(f"问题: {instruction}")
            print(f"答案: {response}")
            df.at[index, 'answer4'] = response
            updated = True

        if updated:
            df.to_csv(output_file_path, index=False, encoding='utf-8')
            print(f"已保存第 {index+1} 行的更新到 {output_file_path}")

if __name__ == "__main__":
    # 原始CSV文件路径
    file_path = './data/LLM/LLM_dataset/Specialized-Abilities-Test-Dataset/Specialized-Abilities-Test-Dataset.csv'

    # 输出CSV文件路径
    output_file_path = './data/LLM/LLM_Metric/Specialized-Abilities-Test-Metric/LLM_Response/GPT-4o.csv'
    directory = os.path.dirname(output_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(output_file_path):
        df = pd.read_csv(file_path)
        df.to_csv(output_file_path, index=False, encoding='utf-8')

    print('-------------')

    # 调用函数处理CSV文件并保存到新的输出文件
    generate_responses_from_csv(output_file_path)
