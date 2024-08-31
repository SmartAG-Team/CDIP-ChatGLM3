import os
import json
import re
from rouge import Rouge  # 使用ROUGE进行文本相似度计算
from openai import OpenAI

# 读取 prompt 文件
with open('./src/prompt.txt', 'r', encoding='utf-8') as prompt_file:
    prompt = prompt_file.read()

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-0e7d0b5cb782481dbff68b487c69d5ca",  # 替换成真实DashScope的API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务endpoint
)

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

# 定义要处理的文件夹路径
folder_path = './data/books/apple/10.1/'
folder_name = os.path.basename(os.path.normpath(folder_path))
# output_file_path = f'./data/fine-tuning_dataset/grape/{folder_name}_output.json'

output_file_path = f'./data/fine-tuning_dataset/apple/apple.json'
output_dir = os.path.dirname(output_file_path)
os.makedirs(output_dir, exist_ok=True)

# 读取文件夹下所有的txt文件
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.txt'):
            file_path = os.path.join(root, filename)
            
            # 读取 context 文件
            with open(file_path, 'r', encoding='utf-8') as data_file:
                context = data_file.read()

            # 生成新的数据
            completion = client.chat.completions.create(
                # model="qwen-long",
                # model="qwen-turbo",
                # model="qwen-plus",
                model="qwen-max-longcontext",
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'system', 'content': context},
                    {'role': 'user', 'content': '从给定的上下文中提出10条多样化的任务指令'}
                ],
                stream=False
            )
            response = completion.choices[0].message.model_dump()

            # 提取生成的内容
            content = response['content']
            try:
                json_data = json.loads(content)
            except json.JSONDecodeError as e:
                json_data = extract_complete_json(content)
                print(json_data)

            # 读取现有数据
            try:
                with open(output_file_path, 'r', encoding='utf-8') as json_file:
                    existing_data = json.load(json_file)
            except FileNotFoundError:
                existing_data = [] 
            except json.JSONDecodeError:
                existing_data = [] 

            # 初始化ROUGE计算器
            rouge = Rouge()

            # 过滤生成的数据，只添加ROUGE-L小于0.7的项，并将文件夹名加入数据中
            new_data_to_add = []
            for new_entry in json_data:
                is_similar = False
                for existing_entry in existing_data:
                    score = rouge.get_scores(str(new_entry), str(existing_entry), avg=True)
                    if score['rouge-l']['f'] >= 0.7:
                        is_similar = True
                        break
                if not is_similar:
                    new_data_to_add.append(new_entry)

            # 添加到现有数据
            if isinstance(existing_data, list):
                existing_data.extend(new_data_to_add)
            else:
                existing_data = [existing_data] + new_data_to_add

            # 保存更新后的数据
            with open(output_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

            # 打印新加的数据数量
            print(f"{folder_name} 中新加了 {len(new_data_to_add)} 个数据")
