from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model = model.to('cuda').eval()
    return model, tokenizer


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Load a specified language model.")
    parser.add_argument("--model_name", type=str, required=True, help="The name or path of the model to load.")

    # 解析命令行参数
    args = parser.parse_args()
    model_name = args.model_name

    # 加载模型
    model, tokenizer = load_model(model_name)

    # 输出模型加载成功的消息
    print(f"Model {model_name} loaded successfully!")
