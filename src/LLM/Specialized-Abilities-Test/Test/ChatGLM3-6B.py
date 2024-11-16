import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_responses_from_csv(output_file_path):
    # 加载预训练模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        r"./model/LLM_models/ChatGLM3_6B", 
        trust_remote_code=True
    )
    gpt_model = AutoModelForCausalLM.from_pretrained(
        r"./model/LLM_models/ChatGLM3_6B", 
        trust_remote_code=True, 
        device='cuda'
    )
    gpt_model = gpt_model.to('cuda').eval()

    # 读取CSV文件
    df = pd.read_csv(output_file_path)

    # 检查是否有 answer, answer1, answer2, answer3 列，没有则创建
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
            response, _ = gpt_model.chat(tokenizer, instruction, history=[])
            response = response.replace("\n", "\\n")  
            print(f"问题: {instruction}")
            print(f"答案: {response}")
            df.at[index, 'answer'] = response
            updated = True

        # 处理 question1 列
        if pd.notna(row['question1']) and pd.isna(row['answer1']):
            instruction = row['question1']
            response, _ = gpt_model.chat(tokenizer, instruction, history=[])
            response = response.replace("\n", "\\n")  
            print(f"问题: {instruction}")
            print(f"生成的答案: {response}")
            df.at[index, 'answer1'] = response
            updated = True

        # 处理 question2 列
        if pd.notna(row['question2']) and pd.isna(row['answer2']):
            instruction = row['question2']
            response, _ = gpt_model.chat(tokenizer, instruction, history=[])
            response = response.replace("\n", "\\n") 
            print(f"问题: {instruction}") 
            print(f"生成的答案: {response}")
            df.at[index, 'answer2'] = response
            updated = True

        # 处理 question3 列
        if pd.notna(row['question3']) and pd.isna(row['answer3']):
            instruction = row['question3']
            response, _ = gpt_model.chat(tokenizer, instruction, history=[])
            response = response.replace("\n", "\\n")  
            print(f"问题: {instruction}")
            print(f"生成的答案: {response}")
            df.at[index, 'answer3'] = response
            updated = True
        
        # 处理 question4 列
        if pd.notna(row['question4']) and pd.isna(row['answer4']):
            instruction = row['question4']
            response, _ = gpt_model.chat(tokenizer, instruction, history=[])
            response = response.replace("\n", "\\n")  
            print(f"问题: {instruction}")
            print(f"生成的答案: {response}")
            df.at[index, 'answer4'] = response
            updated = True

        # 如果有更新，保存整个 DataFrame 到输出文件中，覆盖现有数据
        if updated:
            df.to_csv(output_file_path, index=False, encoding='utf-8')
            print(f"已保存第 {index+1} 行的更新到 {output_file_path}")

if __name__ == "__main__":
    # 原始CSV文件路径
    file_path = './data/LLM/LLM_dataset/Specialized-Abilities-Test-Dataset/Specialized-Abilities-Test-Dataset.csv'

    # 输出CSV文件路径   
    output_file_path = './data/LLM/LLM_Metric/Specialized-Abilities-Test-Metric/LLM_Response/ChatGLM3_6B.csv'
    directory = os.path.dirname(output_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(output_file_path):
        df = pd.read_csv(file_path)
        df.to_csv(output_file_path, index=False, encoding='utf-8')
    
    print('-------------')

    # 调用函数处理CSV文件并保存到新的输出文件
    generate_responses_from_csv(output_file_path)
