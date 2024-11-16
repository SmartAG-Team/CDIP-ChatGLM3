import json
import random

# 假设您的JSON数据存储在一个名为data.json的文件中
json_file_path = './data/13crop.json'
new_json_file_path = './data/13crop_0.1K.json'

# 从文件中读取JSON数据
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 计算要抽取的数据数量（数组长度的一半，向下取整）
half_count = len(data) // 25


# 随机选择一半的数据
selected_data = random.sample(data, half_count)

# 将抽取的数据写入新的JSON文件
with open(new_json_file_path, 'w', encoding='utf-8') as new_file:
    json.dump(selected_data, new_file, ensure_ascii=False, indent=4)

print(f"成功将抽取的数据写入到文件：{new_json_file_path}")
