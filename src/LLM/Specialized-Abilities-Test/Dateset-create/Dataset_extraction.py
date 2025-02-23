import os
import csv
import json
import random

diseases = [
    "苹果链格孢叶斑病", "苹果链格孢病", "苹果黑腐病", "苹果黑腐烂病", "苹果棕色斑点病",
    "苹果灰斑病", "苹果灰色斑点病", "苹果健康", "苹果花叶病", "苹果镶嵌病", "苹果白粉病", "苹果粉霉病",
    "苹果锈病", "苹果锈菌病", 
    "樱桃健康", "樱桃苗白粉病", "樱桃粉霉病", "柑橘健康", "柑橘黄龙病",
    "玉米灰叶斑病", "玉米灰斑病", 
    "玉米灰色叶病", "玉米灰叶病", "玉米健康", "玉米北方叶枯病", "玉米大斑病", "玉米锈病", "玉米锈菌病", 
    "葡萄黑痘病", "葡萄黑斑病", "葡萄黑腐病", "葡萄黑色腐烂病", "葡萄霜霉病", "葡萄毛霉病", 
    "葡萄健康", "葡萄斑枯病", "葡萄叶斑病", "葡萄花叶病毒病", "葡萄花叶病", "葡萄镶嵌病毒病", "葡萄白粉病", 
    "葡萄粉霉病", "葡萄黄化病", "葡萄黄叶病", 
    "桃细菌性斑点病", "桃健康", "辣椒细菌性斑点病",
    "辣椒细菌斑点病", "辣椒细菌性叶斑病", "辣椒健康", "辣椒疮痂病", 
    "辣椒疤痕病",
    "马铃薯健康", "马铃薯晚疫病", "马铃薯晚疫霉病",
    "水稻细菌性叶枯病", "水稻白叶枯病", "水稻褐斑病", "水稻棕色斑点病", "水稻健康", "水稻褐飞虱", "黑尾叶蝉", 
    "水稻叶瘟", "水稻稻瘟病", "水稻东格罗病", "水稻东格罗病毒病", 
    "大豆细菌性叶枯病", "大豆细菌叶枯病", "大豆灰斑病", "大豆尾孢叶枯病", "大豆尾孢叶病", "大豆霜霉病", "大豆毛霉病", 
    "大豆蛙眼病", "大豆眼斑病", "大豆健康", "大豆缺钾症", "大豆钾缺乏症", "大豆锈病", "大豆锈菌病", "大豆靶斑病", 
    "大豆目标斑点病", 
    "草莓健康", "草莓叶烧病", "草莓叶焦病", "草莓高温日灼",
    "番茄细菌性斑点病", "番茄细菌性斑点病", "番茄早疫病", "番茄早期斑点病", "番茄健康", "番茄晚疫病", "番茄晚期斑点病", 
    "番茄叶霉病", "番茄花叶病毒病", "番茄镶嵌病毒病", "番茄斑枯病", "番茄黄化曲叶病毒病", "番茄黄花曲叶病毒病",
    "番茄黄叶卷曲病毒病", 
    "小麦叶锈病", "小麦黄锈病", "小麦条锈病", "小麦褐锈病", "小麦黄锈菌病", "小麦健康", "小麦散黑穗病", "小麦叶枯病"
]

with open("./data/LLM/LLM_dataset/13-Crop-Instruction-Following-Dataset/13crop.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

# 使用集合来跟踪已处理的 instruction，避免重复
seen_instructions = set()
seen_diseases = set()  # 用于存储已提取的病害种类

unique_extracted_data = []
for item in data:
    instruction = item.get("instruction", "")
    for disease in diseases:
        if disease in instruction and instruction not in seen_instructions:
            seen_instructions.add(instruction)
            seen_diseases.add(disease)
            unique_extracted_data.append(item)
            break

# 只保留前200条记录
unique_extracted_data = unique_extracted_data[:200]
random.shuffle(unique_extracted_data)

output_path = './data/LLM/LLM_dataset/Specialized-Abilities-Test-Dataset'
output_file = f'{output_path}/Crop-Disease-200.json'

os.makedirs(output_path, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(unique_extracted_data, file, ensure_ascii=False, indent=4)

print(f"数据已保存到 {output_file}")
print(f"提取了 {len(seen_diseases)} 种病害")
print(f"病害种类: {sorted(seen_diseases)}")

# 读取 JSON 数据
with open(output_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建 CSV 文件
csv_file = './data/LLM/LLM_dataset/Specialized-Abilities-Test-Dataset/disease_prevention_question.csv'

# 构建 CSV 文件，包含表头 question 和 answer
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['disease', 'question', 'answer'])
    
    # 将 JSON 中的数据写入 CSV
    for entry in data:
        disease_name = next((disease for disease in diseases if disease in entry['instruction']), "未找到相关病害")
        writer.writerow([disease_name, entry['instruction'], entry['output']])

print(f'CSV file created: {csv_file}')
