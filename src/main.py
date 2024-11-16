import os
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = r"./model/LLM_models/freeze/freeze10_DMT(1-1)/sft"
TOKENIZER_PATH = r"./model/LLM_models/freeze/freeze10_DMT(1-1)/sft"

# MODEL_PATH = r"./model/LLM_models/chatglm3_6b"
# TOKENIZER_PATH = r"./model/LLM_models/chatglm3_6b"
DEVICE = 'cuda'


# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()

welcome_prompt = "欢迎使用 CDIP-ChatGLM3 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"

def main():
    past_key_values, history = None,[]
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system('cls')
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        # 流式输出回答会返回每一步的结果
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=0.5,
                                                                    temperature=0.5,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            # 只输出每一步新生成的结果
            print(response[current_length:], end="", flush=True)
            current_length = len(response)
        print("")


if __name__ == "__main__":
    main()