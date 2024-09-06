import os
import json
import re
from rouge import Rouge  
from openai import OpenAI
import argparse
import time
import glob


parser = argparse.ArgumentParser(description="Argument parser example")
parser.add_argument('--prompt_dir', type=str, default='./src/create_fine-tuning_dataset/prompt.txt', help='Instruction generation prompt')
parser.add_argument('--prompt2_dir', type=str, default='./src/create_fine-tuning_dataset/prompt2.txt', help='Instruction generation prompt2')

parser.add_argument('--api_key', type=str, default='your api key', help='qwen api key')
parser.add_argument('--folder_path', type=str, default='./data/books/文字数据集/小麦文字分段/小麦文字分段/第三章 小麦病虫害绿色防控技术/', help='Path to the folder to process')
parser.add_argument('--varieties', type=str, default='grape', help='grape/apple/wheat...')

parser.add_argument('--model', type=str, default='qwen-max-longcontext', help='qwen model name')
parser.add_argument('--instruction_number', type=int, default=5, help='Number of instructions proposed per txt file')
parser.add_argument('--delay', type=int, default=5, help='api call intervals')
args = parser.parse_args()


# 读取 prompt 文件
with open(args.prompt_dir, 'r', encoding='utf-8') as prompt_file:
    prompt = prompt_file.read()

with open(args.prompt2_dir, 'r', encoding='utf-8') as prompt_file:
    prompt2 = prompt_file.read()

# 初始化OpenAI客户端
client = OpenAI(
    api_key=args.api_key,  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
)

# 定义要处理的文件夹路径
folder_path = args.folder_path
folder_name = os.path.basename(os.path.normpath(folder_path))
instruction_file_path = f'./data/fine-tuning_dataset/{args.varieties}/{folder_name}_instruction.json'
original_output_file_path = f'./data/fine-tuning_dataset/{args.varieties}/{folder_name}_original_output.json'
output_file_path = f'./data/fine-tuning_dataset/{args.varieties}/{folder_name}_finetune_output.json'
error_log_file='error_log.txt'
# output_file_path = f'./data/fine-tuning_dataset/grape/grape.json'
instruction_dir = os.path.dirname(instruction_file_path)
original_output_dir = os.path.dirname(original_output_file_path)
output_dir = os.path.dirname(output_file_path)
os.makedirs(instruction_dir, exist_ok=True)
os.makedirs(original_output_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


txt_files_content = []
txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

# 遍历每个txt文件，读取内容并存入数组
for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as file:
        content = file.read()
        txt_files_content.append(content)  # 存储此文件夹下所有上下文


def extract_complete_json(content):
    # 定义匹配每个完整JSON对象的正则表达式
    json_object_pattern = r'\{[^{}]*\}'

    # 查找所有完整的JSON对象
    matches = re.findall(json_object_pattern, content)

    # 如果找到匹配项，构建一个完整的JSON数组
    if matches:
        # 构建完整的JSON数组字符串
        complete_json_str = '[' + ','.join(matches) + ']'

        try:
            # 尝试将完整的JSON字符串转换为Python对象
            json_data = json.loads(complete_json_str)
            return json_data
        except json.JSONDecodeError as e:
            print("解析完整的JSON字符串时出错:", e)
            return [{"error": "解析完整的JSON字符串时出错"}]
    else:
        print("未找到完整的JSON对象")
        return [{"error": "未找到完整的JSON对象"}]
    
def split_by_separator(original_output_file_path):
    with open(original_output_file_path, 'r', encoding='utf-8') as json_file:
        original_output = json.load(json_file)

    # 用来存放划分的多个部分
    split_data = []
    current_part = []

    # 遍历original_output，将内容根据"分隔符"分割
    for item in original_output:
        if item['instruction'] == "分隔符":
            # 如果遇到分隔符，将当前部分保存到split_data
            if current_part:
                split_data.append(current_part)
            current_part = []  # 开始新的部分
        else:
            current_part.append(item)

    # 如果最后一部分不为空，添加到split_data
    if current_part:
        split_data.append(current_part)

    return split_data

def instruction_generation(folder_path):  
    for context in txt_files_content:
        # 生成新的数据
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': f'结合上下文提出{args.instruction_number}条指令，请保持标准的JSON格式返回'}
            ],
            stream=False
        )
        response = completion.choices[0].message.model_dump()

        # 提取生成的内容
        content = response['content']
        print(content)

        try:
            json_data = json.loads(content)
            print(json_data)
        except json.JSONDecodeError as e:
            json_data = extract_complete_json(content)
            print(json_data)

        # 读取现有数据
        try:
            with open(instruction_file_path, 'r', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)
        except FileNotFoundError:
            existing_data = [] 
        except json.JSONDecodeError:
            existing_data = [] 

        # 添加到现有数据
        if isinstance(existing_data, list):
            existing_data.extend(json_data)
        else:
            existing_data = [existing_data] + json_data

        # 遍历每个项，补齐"input"和"output"字段
        for entry in existing_data:
            if 'input' not in entry:
                entry['input'] = ""  # 添加input字段
            if 'output' not in entry:
                entry['output'] = ""  # 添加output字段
        
        # 使用 append 添加分隔符
        existing_data.append({
            "instruction": "分隔符",
            "input": "",
            "output": ""
        })

        # 保存更新后的数据
        with open(instruction_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

        # 打印新加的数据数量
        print(f"{folder_name}_instruction.json 中新加了 {len(json_data)} 个数据")
        time.sleep(args.delay)

def original_output(instruction_file_path):
    with open(instruction_file_path, 'r', encoding='utf-8') as json_file:
        instructions = json.load(json_file)
    for item in instructions:
        try:
            if item['instruction'] == "分隔符":
                continue

            instruction = item['instruction']
            
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {'role': 'user', 'content': instruction}
                ],
                stream=False
            )      
            response = completion.choices[0].message.model_dump()

            # 将生成的内容放入output字段
            item['output'] = response['content']
            print(item)
            # 等待一段时间再发送下一个请求
            time.sleep(args.delay)

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    with open(original_output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(instructions, output_file, ensure_ascii=False, indent=4)

    print(f"Original output has been written to {original_output_file_path}")

def output(original_output_file_path):
    with open(original_output_file_path, 'r', encoding='utf-8') as json_file:
        original_output = json.load(json_file)
        split_data = split_by_separator(original_output_file_path)

    co = 0

    for index, data in enumerate(split_data):
        try:
            text = prompt2 + str(data) + '给定的上下文' + '/n' + txt_files_content[index] 
            # print(text)
            # 创建prompt并调用API
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {'role': 'system', 'content': text},
                    {'role': 'user', 'content': '请结合上下文信息针对每一个指令润色一下回复信息，请不要修改原有的结构。'}
                ],
                stream=False
            )
            
            response = completion.choices[0].message.model_dump()
            response_content = response['content']
            # print(response_content)

            # 去除 ```json 和 ``` 标记
            # Remove the text 'json' and extra newline patterns, then strip whitespace
            cleaned_content = re.sub(r'```json|```', '', response_content).strip()     
            try:
                # 解析清理后的内容
                generated_data = json.loads(cleaned_content)
                print(f"Generated Data: {generated_data}")
            except json.JSONDecodeError as e:
                # 解析失败时，将无法解析的内容追加到日志文件中，确保使用 UTF-8 编码
                with open(error_log_file, 'a', encoding='utf-8') as file:
                    file.write(f"Failed to parse JSON content:\n{cleaned_content}\n")
                    file.write(f"Error: {e}\n\n")
                print(cleaned_content)
                print(f"Error parsing JSON response after clean: {e}")
                continue

            # 读取现有的 JSON 文件
            try:
                with open(output_file_path, 'r', encoding='utf-8') as json_file:
                    existing_data = json.load(json_file)
            except FileNotFoundError:
                existing_data = []  # 如果文件不存在，初始化为空列表
            except json.JSONDecodeError:
                existing_data = []  # 如果文件内容为空或无效，初始化为空列表

            # 初始化ROUGE计算器
            rouge = Rouge()

            # 过滤生成的数据，只添加ROUGE-L小于0.7的项，并将文件夹名加入数据中
            new_data_to_add = []
            for new_entry in generated_data:
                is_similar = False
                for existing_entry in existing_data:
                    score = rouge.get_scores(str(new_entry), str(existing_entry), avg=True)
                    if score['rouge-l']['f'] >= 0.7:
                        is_similar = True
                        break
                if not is_similar:
                    new_data_to_add.append(new_entry)
            # 将生成的JSON数据追加到现有数据中
            if isinstance(existing_data, list):
                existing_data.extend(new_data_to_add)
            else:
                existing_data = [existing_data] + new_data_to_add

            # 保存更新后的 JSON 文件
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump(existing_data, output_file, ensure_ascii=False, indent=4)

            print(f"Updated JSON file: {output_file_path} with {len(generated_data)} new entries")
        except Exception as e:
            print(f"An error occurred: {e}")
            break


if __name__ == "__main__":
    instruction_generation(folder_path)
    original_output(instruction_file_path)
    output(original_output_file_path)

   
