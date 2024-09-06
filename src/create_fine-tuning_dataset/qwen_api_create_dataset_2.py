import json
import time
from openai import OpenAI

# 读取 JSON 文件
with open('./data/fine-tuning_dataset/grape/question.json', 'r', encoding='utf-8') as question_file:
    instructions = json.load(question_file)  # 将JSON内容读取为字典列表

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-0e7d0b5cb782481dbff68b487c69d5ca",  # 替换成真实DashScope的API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务endpoint
)

# 设置每次调用之间的延迟（例如1秒）
delay = 1  # 单位：秒

# 遍历每个指令，向模型请求生成回复
for item in instructions:
    try:
        instruction = item['instruction']
        
        # 创建prompt并调用API
        completion = client.chat.completions.create(
            model="qwen-max-longcontext",
            messages=[
                {'role': 'user', 'content': instruction}
            ],
            stream=False
        )
        
        response = completion.choices[0].message.model_dump()

        # 将生成的内容放入output字段
        item['output'] = response['content']
        
        # 等待一段时间再发送下一个请求
        time.sleep(delay)

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# 将修改后的指令集写回JSON文件
with open("output.json", "w", encoding="utf-8") as output_file:
    json.dump(instructions, output_file, ensure_ascii=False, indent=4)

print("Output has been written to output.json")
