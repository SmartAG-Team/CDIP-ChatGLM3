import pandas as pd
import numpy as np
import scipy.stats as stats

files = [
    './data/CV/CV_Metric/CV_results_1.csv',
    './data/CV/CV_Metric/CV_results_2.csv',
    './data/CV/CV_Metric/CV_results_3.csv'
]

dataframes = [pd.read_csv(file) for file in files]
combined_data = pd.concat(dataframes)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Params (M)', 'FPS', 'Latency (ms)']

# 定义列名后缀的映射关系
metric_suffix = {
    'Accuracy': 'Accuracy',
    'Precision': 'Precision',
    'Recall': 'Recall',
    'F1-Score': 'F1',
    'Params (M)': 'Params',
    'FPS': 'FPS',
    'Latency (ms)': 'Latency'
}

summary = combined_data.groupby("Model").agg(
    Mean_Accuracy=('Accuracy', 'mean'),
    Std_Accuracy=('Accuracy', 'std'),
    Mean_Precision=('Precision', 'mean'),
    Std_Precision=('Precision', 'std'),
    Mean_Recall=('Recall', 'mean'),
    Std_Recall=('Recall', 'std'),
    Mean_F1=('F1-Score', 'mean'),
    Std_F1=('F1-Score', 'std'),
    Mean_Params=('Params (M)', 'mean'),
    Std_Params=('Params (M)', 'std'),
    Mean_FPS=('FPS', 'mean'),
    Std_FPS=('FPS', 'std'),
    Mean_Latency=('Latency (ms)', 'mean'),
    Std_Latency=('Latency (ms)', 'std')
)

n = len(files)
t_value = stats.t.ppf(0.975, df=n-1)

# 使用映射关系获取正确的列名后缀
for metric in metrics:
    suffix = metric_suffix[metric]
    std_column = f'Std_{suffix}'
    ci_column = f'CI_{metric}'
    summary[ci_column] = t_value * (summary[std_column] / np.sqrt(n))

# 生成格式化结果
for metric in metrics:
    suffix = metric_suffix[metric]
    mean_column = f'Mean_{suffix}'
    ci_column = f'CI_{metric}'
    summary[f'{metric}_With_CI'] = summary[mean_column].round(4).astype(str) + " ± " + summary[ci_column].round(4).astype(str)

final_summary = summary[[f'{metric}_With_CI' for metric in metrics]]
final_summary.to_csv('./data/CV/CV_Metric/CV_results_sum.csv')

print(final_summary)
