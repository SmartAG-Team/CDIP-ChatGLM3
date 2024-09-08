import json

def check_json_format(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 逐项检查是否缺少必要字段
    for idx, item in enumerate(data):
        missing_fields = []
        for field in ["instruction", "input", "output"]:
            if field not in item:
                missing_fields.append(field)

        if missing_fields:
            print(f"Line {idx*5+2}:")
            print(f"Item {idx} is missing fields: {', '.join(missing_fields)}")     
        # else:
        #     print(f"Item {idx} is correctly formatted.")
    if missing_fields == [] :
        print(f"All Item is correctly formatted.")


if __name__ == "__main__":
    file_path = "./data/fine-tuning_dataset/corn/corn.json"  # 替换为你的JSON文件路径
    check_json_format(file_path)
