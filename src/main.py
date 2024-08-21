import argparse
import json
from model_loader import load_model
from data_loader import load_data
from generate_and_score import generate_responses, calculate_score

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Run the language model scoring pipeline with a specified model.")
    parser.add_argument("--model_name", type=str, required=True, help="The name or path of the model to load.")
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file.')
    parser.add_argument('--data_source', type=str, required=True, help='Path to the source data.')
    parser.add_argument('--output_response', type=str, required=True, help='Path to the output response file.')
    # 解析命令行参数
    args = parser.parse_args()
    model_name = args.model_name
    data_source = args.data_source
    output_csv = args.output_csv
    output_response = args.output_response
    # 加载模型
    model, tokenizer = load_model(model_name)

    # 加载数据
    print("Loading data...")
    data_source, data_reference = load_data(data_source, data_source)

    # 生成回复
    print("Generating response...")
    data_candidate = generate_responses(model, tokenizer, data_source)
    with open(output_response, 'w', encoding='utf-8') as file:
        json.dump(data_candidate, file, ensure_ascii=False, indent=4)

    print(f"Responses have been saved to {output_response}")
    # 计算评分
    print("Calculateing socre...")
    avg_scores = calculate_score(data_candidate, data_reference,output_csv)

    # 输出平均ROUGE F-score
    avg_f_score = (avg_scores["rouge-1"]["f"] + avg_scores["rouge-l"]["f"] + avg_scores["rouge-w"]["f"]) / 3
    print(f"Avg ROUGE F-score: {avg_f_score}")
