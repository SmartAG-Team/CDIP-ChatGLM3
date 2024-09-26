import os
import json

def merge_json_files(folder_path, output_file):
    merged_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            # 读取每个 JSON 文件并将其数据添加到 merged_data 列表中
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                merged_data.extend(data)

    # 将合并后的数据写入到输出文件中
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(merged_data, output, ensure_ascii=False, indent=4)

    print(f"All JSON files in '{folder_path}' have been merged into '{output_file}'.")

if __name__ == "__main__":
    folder_path = "./data/fine-tuning_dataset/grape/finetune_output"  # 替换为你的文件夹路径
    output_file = "./data/fine-tuning_dataset/grape/grape.json"  # 替换为你的输出文件路径
    merge_json_files(folder_path, output_file)
