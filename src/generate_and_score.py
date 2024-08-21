from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
rouge = Rouge()
import csv
def generate_responses(model, tokenizer, data_source):
    responses = []
    for item in data_source:
        instruction = item["instruction"]
        response, _ = model.chat(tokenizer, instruction, history=[])
        entry = {"instruction": instruction, "input": "", "output": response}
        responses.append(entry)

    return responses


def calculate_score(data_candidate, data_reference, output_csv_path):
    # 创建 ROUGE 评估器对象
    rouge = Rouge()

    # 初始化总分字典，包括 BLEU 和 ROUGE
    total_scores = {
        "rouge-1": {"f": 0, "p": 0, "r": 0},
        "rouge-2": {"f": 0, "p": 0, "r": 0},
        "rouge-l": {"f": 0, "p": 0, "r": 0},
        "bleu-4": 0
    }

    # 确保两个数据列表的长度相同
    if len(data_candidate) == len(data_reference):
        # 循环遍历数据列表，提取同个位置的 output
        for item1, item2 in zip(data_candidate, data_reference):
            reference_text = item2.get("output", "")
            generated_text = item1.get("output", "")

            # 计算 ROUGE 指标
            scores = rouge.get_scores(generated_text, reference_text)

            # 调试输出 ROUGE 分数
            print(f"ROUGE scores: {scores}")

            # 累加 ROUGE 指标
            for metric in scores[0]:
                for component in scores[0][metric]:
                    if metric in total_scores and component in total_scores[metric]:
                        total_scores[metric][component] += scores[0][metric][component]

            # 使用 jieba 对文本进行分词
            reference = jieba.lcut(reference_text)
            candidate = jieba.lcut(generated_text)
            # 计算 BLEU-4 分数并累加
            bleu_score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25),
                                       smoothing_function=SmoothingFunction().method1)
            total_scores["bleu-4"] += bleu_score
            print(f'BLEU-4 Score: {bleu_score}')

        # 调试输出总分
        print(f"Total scores: {total_scores}")

        # 计算平均 ROUGE 和 BLEU 分数
        num_samples = len(data_candidate)
        avg_scores = {
            metric: {component: total_scores[metric][component] / num_samples for component in total_scores[metric]} for metric in total_scores if metric != "bleu-4"
        }
        # 检查 avg_scores 的结构
        print(f"avg_scores type: {type(avg_scores)}")  # 应该是 dict
        print(f"avg_scores: {avg_scores}")

        # 检查 avg_scores["rouge-1"] 的结构
        print(f"avg_scores['rouge-1'] type: {type(avg_scores['rouge-1'])}")  # 应该是 dict
        print(f"avg_scores['rouge-1']: {avg_scores['rouge-1']}")

        # 如果 avg_scores["rouge-1"] 是一个元组而不是字典，则说明之前的计算出了问题

        avg_bleu4 = total_scores["bleu-4"] / num_samples

        # 打印平均 ROUGE 和 BLEU 分数
        print("Average ROUGE-1:", avg_scores["rouge-1"])
        print("Average ROUGE-2:", avg_scores["rouge-2"])
        print("Average ROUGE-L:", avg_scores["rouge-l"])
        print(f"Average BLEU-4 Score: {avg_bleu4}")

        # 计算并打印平均 F-score
        avg_f_score = (avg_scores["rouge-1"]["f"] + avg_scores["rouge-2"]["f"] + avg_scores["rouge-l"]["f"]) / 3
        print(f"Average ROUGE F-score: {avg_f_score}")

        # 将分数写入 CSV 文件
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Precision", "Recall", "F-score"])
            writer.writerow(
                ["ROUGE-1", avg_scores["rouge-1"]["p"], avg_scores["rouge-1"]["r"], avg_scores["rouge-1"]["f"]])
            writer.writerow(
                ["ROUGE-2", avg_scores["rouge-2"]["p"], avg_scores["rouge-2"]["r"], avg_scores["rouge-2"]["f"]])
            writer.writerow(
                ["ROUGE-L", avg_scores["rouge-l"]["p"], avg_scores["rouge-l"]["r"], avg_scores["rouge-l"]["f"]])
            writer.writerow(["BLEU-4", "-", "-", avg_bleu4])
            writer.writerow(["Average ROUGE F-score", "-", "-", avg_f_score])

        print(f"Scores have been written to {output_csv_path}")

        # 返回最终分数
        return avg_scores, avg_bleu4, avg_f_score
    else:
        print("数据列表的长度不相同")
        return None