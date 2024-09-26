import json
import random

def split_data(json_file, output_train_file, output_test_file, test_ratio=0.1):
    # 读取原始JSON文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 计算测试数据的数量
    total_count = len(data)
    test_count = int(total_count * test_ratio)
    
    # 随机选择测试数据的索引
    test_indices = random.sample(range(total_count), test_count)
    
    # 分离出测试数据和训练数据
    test_data = [data[i] for i in test_indices]
    train_data = [data[i] for i in range(total_count) if i not in test_indices]
    
    # 将测试数据写入新的JSON文件
    with open(output_test_file, 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    
    # 将剩余的训练数据写入新的JSON文件
    with open(output_train_file, 'w', encoding='utf-8') as file:
        json.dump(train_data, file, ensure_ascii=False, indent=4)
    
    print(f"数据已划分：测试数据 {test_count} 条，训练数据 {total_count - test_count} 条。")


split_data('./data/fine-tuning_dataset/grape/grape.json', './data/fine-tuning_dataset/grape/grape_train_data.json', './data/fine-tuning_dataset/grape/grape_test_data.json')

