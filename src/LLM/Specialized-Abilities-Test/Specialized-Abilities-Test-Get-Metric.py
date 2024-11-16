import os
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 初始化 ROUGE 评分器，计算 ROUGE-1, ROUGE-2, 和 ROUGE-L
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# 定义计算 BLEU-4 和 Average ROUGE F-score 的函数
def calculate_bleu_rouge(candidate, reference):
    # 使用 SmoothingFunction 计算 BLEU-4 指数
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference.split()], candidate.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    # 计算 ROUGE-1, ROUGE-2 和 ROUGE-L 的 F-score
    rouge_scores = scorer.score(reference, candidate)
    rouge_f_scores = [
        rouge_scores['rouge1'].fmeasure,
        rouge_scores['rouge2'].fmeasure,
        rouge_scores['rougeL'].fmeasure
    ]
    # 计算平均的 ROUGE F-score
    avg_rouge_f_score = sum(rouge_f_scores) / len(rouge_f_scores)
    
    return bleu_score, avg_rouge_f_score

# 遍历文件夹，处理所有CSV文件
input_folder = './data/LLM/LLM_Metric/Specialized-Abilities-Test-Metric/LLM_Response'  # 文件夹路径
output_file = './data/LLM/LLM_Metric/Specialized-Abilities-Test-Metric/Specialized-Abilities-Test-Metric.csv'  # 输出汇总文件

# 初始化用于存储汇总结果的列表
summary_results = []

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # 构造完整的文件路径
        file_path = os.path.join(input_folder, filename)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 初始化存储结果的列
        bleu_scores = {f"bleu_{col}": [] for col in ['answer', 'answer1', 'answer2', 'answer3', 'answer4']}
        rouge_scores = {f"rouge_{col}": [] for col in ['answer', 'answer1', 'answer2', 'answer3', 'answer4']}
        
        # 遍历每一行计算BLEU和ROUGE
        for index, row in df.iterrows():
            standard_answer = row['standard_answer']
            
            for col in ['answer', 'answer1', 'answer2', 'answer3', 'answer4']:
                candidate = row[col]
                bleu, avg_rouge = calculate_bleu_rouge(candidate, standard_answer)
                bleu_scores[f"bleu_{col}"].append(bleu)
                rouge_scores[f"rouge_{col}"].append(avg_rouge)
        
        # 计算每一行的 BLEU 和 Average ROUGE F-score 的平均值
        df['average_bleu'] = pd.DataFrame(bleu_scores).mean(axis=1)
        df['average_rouge'] = pd.DataFrame(rouge_scores).mean(axis=1)

        # 计算整体的平均 BLEU 和 ROUGE
        overall_avg_bleu = df['average_bleu'].mean()
        overall_avg_rouge = df['average_rouge'].mean()

        # 将结果添加到汇总结果列表
        summary_results.append({
            'Model': filename.replace('.csv',''),  # 使用文件名作为模型名称
            'BLEU-4(‰) Index': overall_avg_bleu * 1000,
            'Average ROUGE F-score(‰)': overall_avg_rouge * 1000
        })

# 将汇总结果保存到一个CSV文件中
summary_df = pd.DataFrame(summary_results)

# 定义模型的排序顺序
model_order = ['ChatGLM3-6B', 'Qwen-max', 'Llama-3.1-405B-Instruct', 'GPT-4o', 'CDIP-ChatGLM3']

# 按模型顺序重排DataFrame
summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=model_order, ordered=True)
summary_df = summary_df.sort_values('Model')

# 将排序后的结果保存到CSV文件中
summary_df.to_csv(output_file, index=False)

# 输出完成信息
print(f"Summary saved to {output_file}")
