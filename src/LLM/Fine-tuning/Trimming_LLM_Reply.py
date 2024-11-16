import os
import json

# 输入文件夹路径
input_folder = "./data/LLM/LLM_Model_Response/Lora30"

# 遍历输入文件夹下所有JSON文件
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        
        # 读取JSON文件内容
        with open(file_path, mode='r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"文件 {filename} 不是有效的JSON格式，跳过。错误: {e}")
                continue
        
        # 处理数据
        for item in data:
            output = item.get("output", "")
            # 如果 output 是空字典，则取出"name"和"content"字段
            if isinstance(output, dict) and "name" in output and "content" in output:
                name = output.get("name", "")
                content = output.get("content", "")
                # 合并name和content，并更新output字段
                item["output"] = name + content
        
        # 直接覆盖原来的JSON文件
        with open(file_path, mode='w', encoding='utf-8', newline='') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

print("文件处理完成，已覆盖原始文件。")
