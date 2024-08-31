import json
from http import HTTPStatus
import dashscope

# 读取 prompt 和 context 文件内容
with open('./src/prompt.txt', 'r', encoding='utf-8') as prompt_file:
    prompt = prompt_file.read()

# with open('./src/第一节优良鲜食品种.txt', 'r', encoding='utf-8') as data_file:
#     context = data_file.read()
with open('./data/books/grape/第八章 葡萄主要病虫害防控技术/第一节 主要病害防控技术/一 霜霉病.txt', 'r', encoding='utf-8') as data_file:
    context = data_file.read()
dashscope.api_key = "sk-0e7d0b5cb782481dbff68b487c69d5ca"

messages = [
    {
        'role': 'system',
        'content': prompt
    },
    {
        'role': 'system',
        'content': context
    },
    {
        'role': 'user',
        'content': '从给定的上下文中提出一组20条多样化的任务指令'
    }
]

response = dashscope.Generation.call(
    model='llama3.1-405b-instruct',
    messages=messages,
    temperature = 0.7,
    result_format='message',  # 设置结果格式为 "message"
)

if response.status_code == HTTPStatus.OK:
    content = response.output['choices'][0]['message']['content']  # 提取 content
    try:
        # 解析 content 为 JSON 对象
        json_data = json.loads(content)
        # 如果最外层是列表且里面也是列表，则去掉最外层的列表
        if isinstance(json_data, list) and isinstance(json_data[0], list):
            json_data = json_data[0]
    except json.JSONDecodeError as e:
        print("生成的文本不是有效的 JSON 格式:", e)
        json_data = {"error": "生成的文本不是有效的 JSON 格式"}

    file_path = './src/llama_output.json'

    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = []  # 如果文件不存在，初始化为空列表
    except json.JSONDecodeError:
        existing_data = []  # 如果文件内容无效，初始化为空列表

    if isinstance(existing_data, list):
        existing_data.extend(json_data)  # 如果现有数据是列表，使用 extend 方法添加新的数据
    else:
        existing_data = [existing_data] + json_data  # 否则将现有数据和新数据合并成一个列表

    # 保存数据到 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

else:
    print(f'Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}')
