import json

# 读取原始 JSON 文件
input_file_path = './data/LLM/LLM_dataset/General-Dataset-Alpaca/alpaca-chinese-52k.json'
output_file_path = './data/LLM/LLM_dataset/General-Dataset-Alpaca/alpaca_zh_52K.json'

try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"文件 {input_file_path} 未找到")
    data = []
except json.JSONDecodeError as e:
    print(f"读取 JSON 文件时出错: {e}")
    data = []

# 删除每个对象中的指定字段
for item in data:
    item.pop('en_instruction', None)
    item.pop('en_input', None)
    item.pop('en_output', None)

# 将修改后的数据写入新文件
try:
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"处理后的 JSON 数据已保存到 {output_file_path}")
except IOError as e:
    print(f"保存 JSON 文件时出错: {e}")
