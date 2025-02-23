import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator  
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import precision_score, recall_score, f1_score

SRC_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def fig5():
    labels = []
    sizes = []
    directory = './data/LLM/LLM_dataset/13-Crop-Instruction-Following-Dataset/summary'
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            varieties = file
            name = varieties.replace('.json', '').title()  # 将首字母大写
            labels.append(name)
            with open(file_path, "r", encoding="utf-8") as json_file:
                data_source = json.load(json_file)
                sizes.append(len(data_source))

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    colors = sns.color_palette("pastel", len(labels))

    def func(pct, allvals):
        absolute = int(round(pct / 100. * sum(allvals))) 
        return f'{absolute}\n({pct:.1f}%)'  # 使用换行符分隔数值和百分比

    plt.figure(figsize=(3.5, 3.5))  # 设置图形大小
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct=lambda pct: func(pct, sizes), colors=colors, 
                                        textprops=dict(size=8), labeldistance=1.1)

    for i, autotext in enumerate(autotexts):
        angle = (wedges[i].theta1 + wedges[i].theta2) / 2  # 计算扇形的中心角度
        x = 0.75 * np.cos(np.radians(angle))  # 计算x位置
        y = 0.75 * np.sin(np.radians(angle))  # 计算y位置
        autotext.set_position((x, y))  # 设置新的位置
        autotext.set_fontsize(7)

    # 调整标签样式
    plt.setp(texts, size=10) 
    plt.axis('equal') 
    plt.savefig('./fig/Fig5.png', dpi=500)
    plt.show()   

def fig7():
    from scipy import stats
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # 读取三个 CSV 文件
    data1 = pd.read_csv('./data/CV/CV_Metric/CV_results_1.csv')
    data2 = pd.read_csv('./data/CV/CV_Metric/CV_results_2.csv')
    data3 = pd.read_csv('./data/CV/CV_Metric/CV_results_3.csv')

    # 合并三个表格
    combined_data = pd.concat([data1, data2, data3])

    # 确保 CSV 中的模型名称与指定顺序一致
    model_order = [
        'ResNet-34', 'ResNet-50', 'MobileNetV3-Small',
        'EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2',
        'EfficientNetV2-S', 'Swin-transformer-Tiny',
        'FasterNet-T0', 'FasterNet-T1'
    ]
    
    # 检查数据中是否包含所有模型
    if not all(model in combined_data['Model'].values for model in model_order):
        print("Error: Not all models are present in the dataset!")
        return
    
    # 将数值从小数转为百分比
    combined_data[['Accuracy', 'Precision', 'Recall', 'F1-Score']] *= 100

    # 整理数据格式
    melted_data = combined_data.melt(id_vars='Model', 
                                     value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                     var_name='Metric', value_name='Value')

    # 设置模型的顺序
    melted_data['Model'] = pd.Categorical(melted_data['Model'], categories=model_order, ordered=True)
    melted_data = melted_data.sort_values('Model')

    # 计算每个模型和指标的平均值和标准差
    summary_data = melted_data.groupby(['Model', 'Metric']).agg(
        Mean=('Value', 'mean'),
        Std=('Value', 'std')
    ).reset_index()
    summary_data['CI'] = summary_data.apply(
        lambda row: stats.t.interval(0.95, df=len(melted_data[melted_data['Model'] == row['Model']]) - 1, 
                                    loc=row['Mean'], 
                                    scale=row['Std'] / np.sqrt(len(melted_data[melted_data['Model'] == row['Model']])))
        if len(melted_data[melted_data['Model'] == row['Model']]) > 1 else (row['Mean'], row['Mean']), axis=1)

    # 计算误差棒（上限 - 均值）
    summary_data['CI_err'] = summary_data['CI'].apply(lambda ci: ci[1] - ci[0])
    # 创建柱状图，调整宽度和间距
    plt.figure(figsize=(7, 5))

    # 计算柱子的位置，设置间隔
    bar_width = 0.18  # 增大柱子的宽度
    models = summary_data['Model'].unique()
    
    # 确保模型数量是 10
    if len(models) != 10:
        print(f"Error: The number of models is {len(models)}. It should be 10.")
        return
    
    x_indices = np.arange(len(models))  # 每个模型的 x 位置
    offset = np.linspace(-0.3, 0.3, num=4)  # 4个指标的偏移量

    # 使用 Seaborn 的 Set2 配色
    palette = sns.color_palette("Set2")

    # 绘制柱状图
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
        subset = summary_data[summary_data['Metric'] == metric]

        if len(subset) != len(x_indices):
            print(f"Error: Mismatch between number of models ({len(x_indices)}) and data for {metric} ({len(subset)})")
            return

        plt.bar(x_indices + offset[i], subset['Mean'], width=bar_width, label=metric, color=palette[i], zorder=2)

    # 设置 y 轴范围为 80-100
    plt.ylim(85, 100)

    # 添加标题和标签
    plt.xlabel('')
    plt.ylabel('Mean Value (%)')
    plt.xticks(x_indices, models, rotation=45)  # 设置 x 轴标签

    # 设置 y 轴的刻度标签，每隔 2.5 增加一个刻度
    plt.yticks([85 + i * 2.5 for i in range(7)])  # 85 到 100 的刻度


    # 添加网格线和坐标轴，设置更高的 zorder 和透明度
    plt.grid(axis='both', linestyle='-', alpha=0.2, zorder=1)  # 将网格线的 zorder 设置为 1

    # 将 x 和 y 坐标轴设置为 zorder 更高，使其覆盖在柱子上
    plt.gca().spines['left'].set_zorder(4)
    plt.gca().spines['bottom'].set_zorder(4)

    # 在每个指标的最大值上绘制红色五角星
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
        metric_data = summary_data[summary_data['Metric'] == metric]
        max_value = metric_data['Mean'].max()
        max_row = metric_data[metric_data['Mean'] == max_value]

        # 获取当前指标的最大值的 x 坐标
        for idx, row in max_row.iterrows():
            model = row['Model']
            
            # 查找对应柱子的 x 坐标
            bar_x = x_indices[np.where(models == model)[0][0]] + offset[i]  # 获取当前柱子的中心位置
            
            # 计算在柱子上方绘制五角星的位置
            ax = plt.gca()  # 获取当前轴
            
            # 绘制五角星
            ax.plot(bar_x, max_value + 0.6, '*', color='black', markersize=6, zorder=5)

    # 设置图例位置和格式
    plt.legend(title='', bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=4)

    # 设置 X 轴和 Y 轴刻度字体
    plt.setp(plt.gca().get_xticklabels(), fontname='Times New Roman', fontsize=10)
    plt.setp(plt.gca().get_yticklabels(), fontname='Times New Roman', fontsize=10)

    # 保存图形，设置 DPI 为 300
    plt.tight_layout()
    plt.savefig('./fig/Fig7.png', dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    plt.show()


def fig8():
   # Load data from CSV
    data = pd.read_csv(r'./data/CV/CV_Metric/CV_confusion_matrix.csv', index_col=0)

    # Calculate accuracies for each disease
    accuracies = data.apply(lambda x: x[x.name] / x.sum() * 100 if x.sum() > 0 else 0)

    # 提取作物类别和病害名称并为每个作物类别分配颜色
    def get_crop_disease(label):
        parts = label.split()
        crop = parts[0]
        disease = ' '.join(parts[1:])
        return crop, disease

    # 输出每个类别的精度
    for label, accuracy in accuracies.items():
        crop, disease = get_crop_disease(label)
        print(f"{crop} - {disease}: {accuracy:.2f}%")

    # 获取作物名称和病害名称
    crops_diseases = accuracies.index.to_series().apply(get_crop_disease)
    crops = crops_diseases.apply(lambda x: x[0])  # 提取作物名称
    unique_crops = crops.unique()

    # 使用 Seaborn 的 Set2 配色并补充其他颜色到 13 种
    set2_palette = sns.color_palette("Set2", n_colors=8)  # Seaborn Set2 提供的 8 种颜色
    extra_palette = sns.color_palette("Paired", n_colors=5)  # 从 Paired 中补充 5 种颜色
    palette = set2_palette + extra_palette  # 合并两种调色板

    # 创建颜色映射
    color_map = {crop: palette[i % 13] for i, crop in enumerate(unique_crops)}

    # 按照作物和病害的字母顺序重新排序
    accuracies = accuracies.loc[sorted(accuracies.index, key=lambda x: (get_crop_disease(x)[0], get_crop_disease(x)[1]))]

    # 为每个疾病分配颜色
    colors = [color_map[get_crop_disease(label)[0]] for label in accuracies.index]

    # Set Seaborn style
    sns.set(style='whitegrid')

    # Create a horizontal bar chart
    plt.figure(figsize=(5.96, 9.5))
    plt.barh(accuracies.index, accuracies, color=colors)
    plt.title('', fontdict={'fontsize': 10, 'fontname': 'Times New Roman'})
    plt.xlabel('Accuracy (%)', fontdict={'fontsize': 10, 'fontname': 'Times New Roman'})
    plt.ylabel('', fontdict={'fontsize': 10, 'fontname': 'Times New Roman'})
    plt.xlim(60, 100)  # Set x-axis range to 60-100
    plt.grid(axis='x')
    plt.xticks(fontsize=10, fontname='Times New Roman')  # Set x-ticks font
    plt.yticks(fontsize=10, fontname='Times New Roman')  # Set y-ticks font
    original_labels = accuracies.index.tolist()
    modified_labels = [" ".join([label.split()[1].capitalize()] + label.split()[2:]) for label in original_labels]

    plt.gca().set_yticklabels(modified_labels)
    # Invert y-axis to reverse the order
    plt.gca().invert_yaxis()

    # Remove top and bottom margins
    plt.gca().margins(y=0)  # 去掉柱状图的上下空白

    # Add a legend for the crops, placing it at the top of the plot with two rows
    handles = [plt.Line2D([0], [0], color=color_map[crop], lw=4) for crop in unique_crops]
    plt.legend(
        handles, 
        unique_crops, 
        bbox_to_anchor=(0.37, 1), 
        loc='lower center', 
        ncol=5,
        prop={'family': 'Times New Roman', 'size': 10}  # 设置字体为 Times New Roman，大小为 10
    )

    plt.tight_layout()

    # Save the plot with 300 DPI
    plt.savefig(r"fig/Fig8.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def fig9():
    # Set global font properties for the plots
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # Load the data from CSV files
    df_cmmlu = pd.read_csv('./data/LLM/LLM_Metric/CMMLU-Metric/CMMLU_Metric.csv')
    metrics_df = pd.read_csv('./data/LLM/LLM_Metric/Fine-tuning-Metric/Fine-tuning-Metric.csv')
    error_df = pd.read_csv('./data/LLM/LLM_Metric/Error-rate-Metric/Error-rate-Metric-token.csv')

    # Filter the CMMLU dataset for 'Overall' subject results only
    df_cmmlu = df_cmmlu[df_cmmlu['Subject'] == 'Overall']

    # Remove unwanted models from the dataset
    models_to_remove = [
        'Freeze10-MTL(S2.5K+G1.25K)',
        'Freeze10-MTL(S2.5K+G2.5K)',
        'Freeze10-MTL(S2.5K+G5K)',
        'Freeze10-MTL(S2.5K+G10K)',
        'Freeze10-MTL(S2.5K+G20K)',
        'Freeze10(S2.5K)-DMT(S0.1K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S0.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S1.25K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G5K)'
    ]

    df_cmmlu = df_cmmlu[~df_cmmlu['Model'].isin(models_to_remove)]

    # Define the order of models for consistency across the plots
    model_order = [
        'ChatGLM3_6B','Lora3', 'Lora5', 'Lora10', 'Lora15', 'Lora20', 'Lora30',
        'Freeze3','Freeze5','Freeze10', 'Freeze15', 'Freeze20', 'Freeze30'
    ]
    df_cmmlu['Model'] = pd.Categorical(df_cmmlu['Model'], categories=model_order, ordered=True)
    df_cmmlu = df_cmmlu.sort_values('Model')

    # Filter the metrics and error datasets to include only the models in model_order
    metrics_df_filtered = metrics_df[metrics_df['Crop'] == 'average']
    metrics_df_filtered = metrics_df_filtered[metrics_df['Model'].isin(model_order)].drop_duplicates(subset=['Model'])
    metrics_df_filtered['Model'] = pd.Categorical(metrics_df_filtered['Model'], categories=model_order, ordered=True)
    metrics_df_filtered = metrics_df_filtered.sort_values('Model').reset_index(drop=True)

    # Define two ranges of models to group metrics
    models_in_range1 = [
        'ChatGLM3_6B', 'Lora3', 'Lora5', 'Lora10', 'Lora15', 'Lora20', 'Lora30'
    ]
    models_in_range2 = [
        'ChatGLM3_6B', 'Freeze3', 'Freeze5', 'Freeze10', 'Freeze15', 'Freeze20', 'Freeze30'
    ]

    # Split metrics data into two groups based on model ranges
    metrics_df_filtered1 = metrics_df_filtered[metrics_df_filtered['Model'].isin(models_in_range1)]
    metrics_df_filtered1['Model'] = pd.Categorical(metrics_df_filtered1['Model'], categories=models_in_range1, ordered=True)
    metrics_df_filtered1 = metrics_df_filtered1.sort_values('Model').reset_index(drop=True)

    metrics_df_filtered2 = metrics_df_filtered[metrics_df_filtered['Model'].isin(models_in_range2)]
    metrics_df_filtered2['Model'] = pd.Categorical(metrics_df_filtered2['Model'], categories=models_in_range2, ordered=True)
    metrics_df_filtered2 = metrics_df_filtered2.sort_values('Model').reset_index(drop=True)

    # Filter and organize error data in the same way
    error_df_filtered = error_df[error_df['Model'].isin(model_order)].set_index('Model').loc[model_order]
    error_df_filtered1 = error_df_filtered.loc[models_in_range1]
    error_df_filtered2 = error_df_filtered.loc[models_in_range2]

    # Split CMMLU data into two groups based on model ranges
    df_cmmlu_filtered1 = df_cmmlu[df_cmmlu['Model'].isin(models_in_range1)]
    df_cmmlu_filtered1['Model'] = pd.Categorical(df_cmmlu_filtered1['Model'], categories=models_in_range1, ordered=True)
    df_cmmlu_filtered1 = df_cmmlu_filtered1.sort_values('Model')

    df_cmmlu_filtered2 = df_cmmlu[df_cmmlu['Model'].isin(models_in_range2)]
    df_cmmlu_filtered2['Model'] = pd.Categorical(df_cmmlu_filtered2['Model'], categories=models_in_range2, ordered=True)
    df_cmmlu_filtered2 = df_cmmlu_filtered2.sort_values('Model')

    # Create a figure with four subplots; two rows and two columns
    fig = plt.figure(figsize=(7, 8))  # Adjust width to fit side-by-side subplots

    # Create a grid layout; first two subplots side by side, next two subplots below
    grid = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    # Adjust the spacing between subplots
    grid.update(wspace=0.05, hspace=0.05)

    # Left subplot (models_in_range1)
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.set_title('(a)', fontsize=14, fontweight='bold', y=0.85)
    ax1.set_ylabel('Values of BLEU-4 and ROUGE (‰)')
    bleu_line1, = ax1.plot(
        metrics_df_filtered1['Model'], metrics_df_filtered1['BLEU-4(‰)'],
        marker='o', label='BLEU-4 Index', color='#1f77b4'
    )
    rouge_line1, = ax1.plot(
        metrics_df_filtered1['Model'], metrics_df_filtered1['Average ROUGE F-score(‰)'],
        marker='s', label='Average ROUGE F-score', color='#ffdd44'
    )
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(50, 450)

    # Create a secondary y-axis for Error Rate
    ax1_err = ax1.twinx()


    error_line1, = ax1_err.plot(
        metrics_df_filtered1['Model'], error_df_filtered1['Error Rate (%)'],
        marker='^', label='Error Rate', color='red'
    )
    # Hide the labels and ticks of the secondary y-axis on ax1
    ax1_err.set_ylabel('')
    ax1_err.set_ylim(0, 4)  # Set y-axis limits from 0 to 4
    ax1_err.set_yticklabels([])
    ax1_err.tick_params(axis='y', which='both', length=0)

    # Remove x-axis labels on ax1

    # Right subplot (models_in_range2)
    ax2 = fig.add_subplot(grid[0, 1], sharey=ax1)  # Share y-axis with ax1
    ax2.set_title('(b)', fontsize=14, fontweight='bold', y=0.85)
    bleu_line2, = ax2.plot(
        metrics_df_filtered2['Model'], metrics_df_filtered2['BLEU-4(‰)'],
        marker='o', label='BLEU-4 Index', color='#1f77b4'
    )
    rouge_line2, = ax2.plot(
        metrics_df_filtered2['Model'], metrics_df_filtered2['Average ROUGE F-score(‰)'],
        marker='s', label='Average ROUGE F-score', color='#ffdd44'
    )
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(50, 450)
    ax2.tick_params(axis='y', labelleft=False)  # Hide y-axis labels on ax2

    # Create a secondary y-axis for Error Rate on ax2

    ax2_err = ax2.twinx()

    
    error_line2, = ax2_err.plot(
        metrics_df_filtered2['Model'], error_df_filtered2['Error Rate (%)'],
        marker='^', label='Error Rate', color='red'
    )
    ax2_err.set_ylabel('Error Rate (%)', labelpad=5)
    ax2_err.set_ylim(0, 4)  # Set y-axis limits from 0 to 4
    ax2_err.yaxis.set_major_locator(MultipleLocator(0.5))  # Set tick intervals of 0.5

    lines = [bleu_line1, rouge_line1, error_line1]
    labels = [line.get_label() for line in lines]
    
    # Third and fourth subplots, below the first two, showing bar plots


    ax3 = fig.add_subplot(grid[1, 0], sharex = ax1 )  # 下排共享 x 轴
    ax4 = fig.add_subplot(grid[1, 1], sharex=ax2, sharey=ax3 )  # 右

    # Remove x-axis labels on ax1 and ax2
    ax1.tick_params(labelbottom=False)  # 仅移除 ax1 的 x 轴标签
    ax2.tick_params(labelbottom=False)  # 仅移除 ax2 的 x 轴标签


    sns.set_theme(style="whitegrid")
    set2_palette = sns.color_palette('Set2', n_colors=8)
    husl_palette = sns.color_palette('husl', n_colors=3)
    palette = set2_palette + husl_palette

    sns.barplot(
        data=df_cmmlu_filtered1, x='Model', y='Accuracy', hue='Shot', palette=palette,
        hue_order=[0, 5], ax=ax3, zorder=2
    )
    sns.barplot(
        data=df_cmmlu_filtered2, x='Model', y='Accuracy', hue='Shot', palette=palette,
        hue_order=[0, 5], ax=ax4, zorder=2
    )

    # Configure y-axis and labels for ax3 and ax4
    ax3.set_ylim((45, 55))
    ax4.set_ylim((45, 55))
    ax3.set_ylabel('CMMLU Accuracy (%)')
    ax4.set_ylabel('')
    ax4.tick_params(axis='y', labelleft=False)  # Hide y-axis labels on ax4
    ax3.set_xlabel('')
    ax4.set_xlabel('')
    ax3.tick_params(axis='x', rotation=45)
    ax4.tick_params(axis='x', rotation=45)
    ax3.set_title('(c)', fontsize=14, fontweight='bold', y=0.85)
    ax4.set_title('(d)', fontsize=14, fontweight='bold', y=0.85)

    # Set custom x-tick labels
    model_order_display_ax3 = [
        'ChatGLM3-6B', 'LoRA3', 'LoRA5', 'LoRA10', 'LoRA15', 'LoRA20', 'LoRA30'
    ]
    model_order_display_ax4 = [
        'ChatGLM3-6B', 'Freeze3', 'Freeze5', 'Freeze10', 'Freeze15', 'Freeze20', 'Freeze30'
    ]

    ax3.set_xticklabels(model_order_display_ax3, rotation=45, ha='right')
    ax4.set_xticklabels(model_order_display_ax4, rotation=45, ha='right')

    # Remove legends from ax3 and ax4
    handles_bar, labels_bar = ax3.get_legend_handles_labels()
    ax3.get_legend().remove()
    ax4.get_legend().remove()

    # Combine legends for all plots
    lines = [bleu_line1, rouge_line1, error_line1]
    labels = [line.get_label() for line in lines]

    fig.legend(
        lines, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.95),frameon=False,
        prop={'family': 'Times New Roman', 'size': 10}
    )
    fig.legend(
        handles_bar, ['Zero-shot', 'Five-shot'], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.92),frameon=False,
        prop={'family': 'Times New Roman', 'size': 10}
    )

    # Add grid lines
    ax1.grid(True, zorder=1)
    ax2.grid(True, zorder=1)
    ax3.grid(True, zorder=1)
    ax4.grid(True, zorder=1)

    # Adjust layout to prevent the legend from overlapping with subplots
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the figure and display
    plt.savefig('./fig/Fig9.png', dpi=500, bbox_inches='tight')
    plt.show()    

def fig10():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # Load the data from CSV files
    df_cmmlu = pd.read_csv('./data/LLM/LLM_Metric/CMMLU-Metric/CMMLU_Metric.csv')
    metrics_df = pd.read_csv('./data/LLM/LLM_Metric/Fine-tuning-Metric/Fine-tuning-Metric.csv')
    error_df = pd.read_csv('./data/LLM/LLM_Metric/Error-rate-Metric/Error-rate-Metric-token.csv')

    # Filter the CMMLU dataset for 'Overall' subject results only
    df_cmmlu = df_cmmlu[df_cmmlu['Subject'] == 'Overall']

    # Remove unwanted models from the dataset
    models_to_remove = [
        'ChatGLM3_6B','Lora3', 'Lora5', 'Lora10', 'Lora15', 'Lora20', 'Lora30',
        'Freeze3','Freeze5', 'Freeze15', 'Freeze20', 'Freeze30'
    ]
    df_cmmlu = df_cmmlu[~df_cmmlu['Model'].isin(models_to_remove)]

    # Define the order of models for consistency across the plots
    model_order = [
        'Freeze10',
        'Freeze10-MTL(S2.5K+G1.25K)',
        'Freeze10-MTL(S2.5K+G2.5K)',
        'Freeze10-MTL(S2.5K+G5K)',
        'Freeze10-MTL(S2.5K+G10K)',
        'Freeze10-MTL(S2.5K+G20K)',
        'Freeze10(S2.5K)-DMT(S0.1K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S0.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S1.25K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G5K)'
    ]
    df_cmmlu['Model'] = pd.Categorical(df_cmmlu['Model'], categories=model_order, ordered=True)
    df_cmmlu = df_cmmlu.sort_values('Model')

    # Filter the metrics and error datasets to include only the models in model_order
    metrics_df_filtered = metrics_df[metrics_df['Crop'] == 'average']
    metrics_df_filtered = metrics_df_filtered[metrics_df['Model'].isin(model_order)].drop_duplicates(subset=['Model'])
    metrics_df_filtered['Model'] = pd.Categorical(metrics_df_filtered['Model'], categories=model_order, ordered=True)
    metrics_df_filtered = metrics_df_filtered.sort_values('Model').reset_index(drop=True)

    # Define two ranges of models to group metrics
    models_in_range1 = [
        'Freeze10',
        'Freeze10-MTL(S2.5K+G1.25K)',
        'Freeze10-MTL(S2.5K+G2.5K)',
        'Freeze10-MTL(S2.5K+G5K)',
        'Freeze10-MTL(S2.5K+G10K)',
        'Freeze10-MTL(S2.5K+G20K)'
    ]
    models_in_range2 = [
        'Freeze10',
        'Freeze10(S2.5K)-DMT(S0.1K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S0.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S1.25K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G5K)'
    ]

    # Split metrics data into two groups based on model ranges
    metrics_df_filtered1 = metrics_df_filtered[metrics_df_filtered['Model'].isin(models_in_range1)]
    metrics_df_filtered1['Model'] = pd.Categorical(metrics_df_filtered1['Model'], categories=models_in_range1, ordered=True)
    metrics_df_filtered1 = metrics_df_filtered1.sort_values('Model').reset_index(drop=True)

    metrics_df_filtered2 = metrics_df_filtered[metrics_df_filtered['Model'].isin(models_in_range2)]
    metrics_df_filtered2['Model'] = pd.Categorical(metrics_df_filtered2['Model'], categories=models_in_range2, ordered=True)
    metrics_df_filtered2 = metrics_df_filtered2.sort_values('Model').reset_index(drop=True)
    # Filter and organize error data in the same way
    error_df_filtered = error_df[error_df['Model'].isin(model_order)].set_index('Model').loc[model_order]
    error_df_filtered1 = error_df_filtered.loc[models_in_range1]
    error_df_filtered2 = error_df_filtered.loc[models_in_range2]

    # Split CMMLU data into two groups based on model ranges
    df_cmmlu_filtered1 = df_cmmlu[df_cmmlu['Model'].isin(models_in_range1)]
    df_cmmlu_filtered1['Model'] = pd.Categorical(df_cmmlu_filtered1['Model'], categories=models_in_range1, ordered=True)
    df_cmmlu_filtered1 = df_cmmlu_filtered1.sort_values('Model')

    df_cmmlu_filtered2 = df_cmmlu[df_cmmlu['Model'].isin(models_in_range2)]
    df_cmmlu_filtered2['Model'] = pd.Categorical(df_cmmlu_filtered2['Model'], categories=models_in_range2, ordered=True)
    df_cmmlu_filtered2 = df_cmmlu_filtered2.sort_values('Model')


    # Create a figure with three subplots; first two subplots are side by side, third is below
    fig = plt.figure(figsize=(7, 8))  # Adjust width to fit side-by-side subplots

    # Create a grid layout; first two subplots side by side
    grid = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    # Adjust the spacing between subplots
    grid.update(wspace=0.05, hspace=0.05)

    # Left subplot (models_in_range1)
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.set_title('(a)', fontsize=14, fontweight='bold', y=0.85)
    ax1.set_ylabel('Values of BLEU-4 and ROUGE (‰)')
    bleu_line1, = ax1.plot(
        metrics_df_filtered1['Model'], metrics_df_filtered1['BLEU-4(‰)'], 
        marker='o', label='BLEU-4 Index', color='#1f77b4'
    )
    rouge_line1, = ax1.plot(
        metrics_df_filtered1['Model'], metrics_df_filtered1['Average ROUGE F-score(‰)'], 
        marker='s', label='Average ROUGE F-score', color='#ffdd44'
    )
    ax1.tick_params(axis='x', rotation=0)
    ax1.set_ylim(200, 450)

    # Create a secondary y-axis for Error Rate
    ax1_err = ax1.twinx()
    error_line1, = ax1_err.plot(
        metrics_df_filtered1['Model'], error_df_filtered1['Error Rate (%)'], 
        marker='^', label='Error Rate', color='red'
    )
    # Hide the labels and ticks of the secondary y-axis on ax1
    ax1_err.set_ylabel('')
    ax1_err.set_yticklabels([])
    ax1_err.tick_params(axis='y', which='both', length=0)

    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    ax1_err.set_ylim(0, 1.5)  
    # Right subplot (models_in_range2)
    ax2 = fig.add_subplot(grid[0, 1], sharey=ax1)  # Share y-axis with ax1
    ax2.set_title('(b)', fontsize=14, fontweight='bold', y=0.85)
    # ax2.set_ylabel('Values of BLEU-4 and ROUGE (‰)')  # Already set on ax1
    bleu_line2, = ax2.plot(
        metrics_df_filtered2['Model'], metrics_df_filtered2['BLEU-4(‰)'], 
        marker='o', label='BLEU-4 Index', color='#1f77b4'
    )
    rouge_line2, = ax2.plot(
        metrics_df_filtered2['Model'], metrics_df_filtered2['Average ROUGE F-score(‰)'], 
        marker='s', label='Average ROUGE F-score', color='#ffdd44'
    )
    ax2.tick_params(axis='x', rotation=0)
    ax2.set_ylim(200, 450)
    ax2.tick_params(axis='y', labelleft=False)  # Hide y-axis labels on ax2

    # Create a secondary y-axis for Error Rate on ax2
    ax2_err = ax2.twinx()
    error_line2, = ax2_err.plot(
        metrics_df_filtered2['Model'], error_df_filtered2['Error Rate (%)'], 
        marker='^', label='Error Rate', color='red'
    )
    ax2_err.set_ylabel('Error Rate (%)', labelpad=5)
    ax2_err.set_ylim(0, 1.5)  
    ax2_err.yaxis.set_major_locator(MultipleLocator(0.3))  # Set tick intervals of 0.5
    # Combine legends for ax1 and ax2, place it above the two subplots
    lines = [bleu_line1, rouge_line1, error_line1]
    labels = [line.get_label() for line in lines]

    # Third subplot, below the first two, showing a bar plot
    ax3 = fig.add_subplot(grid[1, 0], sharex = ax1 )  # 下排共享 x 轴
    ax4 = fig.add_subplot(grid[1, 1], sharex=ax2, sharey=ax3 )  # 右

    # Remove x-axis labels on ax1 and ax2
    ax1.tick_params(labelbottom=False)  # 仅移除 ax1 的 x 轴标签
    ax2.tick_params(labelbottom=False)  # 仅移除 ax2 的 x 轴标签

    sns.set_theme(style="whitegrid")
    set2_palette = sns.color_palette('Set2', n_colors=8)
    husl_palette = sns.color_palette('husl', n_colors=3)
    palette = set2_palette + husl_palette

    sns.barplot(
        data=df_cmmlu_filtered1, x='Model', y='Accuracy', hue='Shot', palette=palette,
        hue_order=[0, 5], ax=ax3, zorder=2
    )

    sns.barplot(
        data=df_cmmlu_filtered2, x='Model', y='Accuracy', hue='Shot', palette=palette,
        hue_order=[0, 5], ax=ax4, zorder=2
    )
    # Configure y-axis and labels for ax3
    ax3.set_ylim((45, 55))
    ax4.set_ylim((45, 55))
    ax3.set_ylabel('CMMLU Accuracy (%)')
    ax4.set_ylabel('')
    ax4.tick_params(axis='y', labelleft=False)  # Hide y-axis labels on ax4
    ax3.set_xlabel('')
    ax4.set_xlabel('')
    ax3.tick_params(axis='x', rotation=45)
    ax4.tick_params(axis='x', rotation=45)
    ax3.set_title('(c)', fontsize=14, fontweight='bold', y=0.85)
    ax4.set_title('(d)', fontsize=14, fontweight='bold', y=0.85)

    model_order_display_ax3 = [
        'Freeze10(S:2.5K)',
        'Freeze10-MTL\n(S:2.5K+G:1.25K)',
        'Freeze10-MTL\n(S:2.5K+G:2.5K)',
        'Freeze10-MTL\n(S:2.5K+G:5K)',
        'Freeze10-MTL\n(S:2.5K+G:10K)',
        'Freeze10-MTL\n(S:2.5K+G:20K)'
    ]
    model_order_display_ax4 = [
        'Freeze10(S:2.5K)',
        'Freeze10(S:2.5K)-\nDMT(S:0.1K+G:2.5K)',
        'Freeze10(S:2.5K)-\nDMT(S:0.5K+G:2.5K)',
        'Freeze10(S:2.5K)-\nDMT(S:1.25K+G:2.5K)',
        'Freeze10(S:2.5K)-\nDMT(S:2.5K+G:2.5K)',
        'Freeze10(S:2.5K)-\nDMT(S:2.5K+G:5K)'
    ]
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels(model_order_display_ax3, rotation=90, ha='center')
    ax4.set_xticklabels(model_order_display_ax4, rotation=90, ha='center')

    handles_bar, labels_bar = ax3.get_legend_handles_labels()
    ax3.get_legend().remove()
    ax4.get_legend().remove()
    # Combine legends for all plots
    lines = [bleu_line1, rouge_line1, error_line1]
    labels = [line.get_label() for line in lines]

    all_handles = lines + handles_bar
    all_labels = labels + ['Zero-shot', 'Five-shot']

    fig.legend(
        lines, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.95),frameon=False,
        prop={'family': 'Times New Roman', 'size': 10}
    )
    fig.legend(
        handles_bar, ['Zero-shot', 'Five-shot'], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.92),frameon=False,
        prop={'family': 'Times New Roman', 'size': 10}
    )

        # Add grid lines
    ax1.grid(True, zorder=1)
    ax2.grid(True, zorder=1)
    ax3.grid(True, zorder=1)
    ax4.grid(True, zorder=1)

    # Adjust layout to prevent the legend from overlapping with subplots
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(bottom=0.2)  # 增加底部边距以容纳标签

    # Save the figure and display
    plt.savefig('./fig/Fig10.png', dpi=500, bbox_inches='tight')
    plt.show()

def fig11():
    save_path='./fig/Fig11.png'
    dpi=500
    df = pd.read_csv('./data/LLM/LLM_Metric/Specialized-Abilities-Test-Metric/Specialized-Abilities-Test-Metric.csv')
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 重置 Seaborn 样式
    sns.reset_orig()
    palette = sns.color_palette('Set2', n_colors=8)

    # 创建画布
    fig, axes = plt.subplots(2, 1, figsize=(6, 4))

    # 第一个子图：断轴效果
    ax_main = axes[0]
    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", size=1.0, pad=0, sharex=ax_main)
    ax_top.set_ylim(99, 108)
    ax_top.spines['bottom'].set_visible(False)  # 隐藏 ax_top 的下边框
    ax_top.tick_params(labelbottom=False)

    ax_bottom = ax_main
    ax_bottom.set_ylim(0, 5)
    ax_bottom.spines['top'].set_visible(False)  # 隐藏 ax_bottom 的上边框

    # 添加断轴符号
    d = .010
    kwargs_top = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (0.1-d,0.1+d), **kwargs_top)
    ax_top.plot((1 - d, 1 + d), (0.1-d, 0.1+d), **kwargs_top)

    kwargs_bottom = dict(transform=ax_bottom.transAxes, color='k', clip_on=False)
    ax_bottom.plot((-d, +d), (0.8 - d, 0.8 + d), **kwargs_bottom)
    ax_bottom.plot((1 - d, 1 + d), (0.8 - d, 0.8 + d), **kwargs_bottom)

    # 绘制柱状图
    top_plot = sns.barplot(x='Model', y='BLEU-4(‰) Index', data=df, ax=ax_top, color=palette[0], width=0.45, zorder=3)
    bottom_plot = sns.barplot(x='Model', y='BLEU-4(‰) Index', data=df, ax=ax_bottom, color=palette[0], width=0.45, zorder=3)

    # 在柱子上方显示数值
    for p in top_plot.patches:
        top_plot.annotate(format(p.get_height(), '.2f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'center', 
                         xytext = (0, 3), textcoords = 'offset points',
                         fontproperties=font_manager.FontProperties(family='Times New Roman', size=10))
    
    for p in bottom_plot.patches:
        bottom_plot.annotate(format(p.get_height(), '.2f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center', 
                            xytext = (0, 3), textcoords = 'offset points',
                            fontproperties=font_manager.FontProperties(family='Times New Roman', size=10))

    # 设置 Y 轴标签居中显示
    fig.text(0.03, 0.78, 'BLEU-4(‰) Index', fontsize=10, va='center', rotation='vertical',
             fontdict={'family': 'Times New Roman'})
    ax_bottom.set_xticklabels([])
    ax_bottom.set_xlabel('')

    # 第二个子图：绘制 Average ROUGE F-score
    ax2 = axes[1]
    rouge_plot = sns.barplot(x='Model', y='Average ROUGE F-score(‰)', data=df, ax=ax2, color=palette[1], width=0.45, zorder=3)

    # 在柱子上方显示数值
    for p in rouge_plot.patches:
        rouge_plot.annotate(format(p.get_height(), '.2f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center',
                            xytext = (0, 3), textcoords = 'offset points',
                            fontproperties=font_manager.FontProperties(family='Times New Roman', size=10))

    ax2.set_xlabel('', fontsize=10)
    ax2.set_ylabel('Average ROUGE F-score(‰)', fontsize=10, fontdict={'family': 'Times New Roman'})

    # 设置字体属性
    font_properties = font_manager.FontProperties(family='Times New Roman', weight='normal', style='normal', size=10)

    # 设置 Y 轴刻度字体
    for ax in [ax_top, ax_bottom, ax2]:
        ax.set_yticklabels(ax.get_yticks(), fontproperties=font_properties)

    # 设置 X 轴刻度
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=10, fontdict={'family': 'Times New Roman'})
    ax2.tick_params(axis='x', rotation=0)

    # 设置 Y 轴范围
    ax2.set_ylim(0, 520)

    # 设置网格线和层级
    ax_top.grid(True, zorder=1)
    ax_bottom.grid(True, zorder=1)
    ax2.grid(True, zorder=1)

    # 隐藏 ax_top 的 X 轴刻度线
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False)
    ax_top.set_ylabel('')
    ax_bottom.set_ylabel('')

    # 设置 Y 轴刻度为整数并去掉小数点
    for ax in [ax_top, ax_bottom, ax2]:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # 设置特定 Y 轴刻度的间隔
    ax2.yaxis.set_major_locator(MultipleLocator(200))
    ax_top.yaxis.set_major_locator(MultipleLocator(4))
    ax_bottom.yaxis.set_major_locator(MultipleLocator(4))
    ax_top.set_title('(a)', fontsize=14, fontweight='bold', fontname='Times New Roman', y=0.70)
    ax2.set_title('(b)', fontsize=14, fontweight='bold', fontname='Times New Roman', y=0.80)


    # 调整布局
    plt.tight_layout()

    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path, dpi=dpi)

    # 显示图形
    plt.show()

def sfig1():
    import matplotlib
    matplotlib.use('TkAgg')
    # 读取CSV文件
    data = pd.read_csv('./data/LLM/LLM_Metric/Fine-tuning-Metric/Fine-tuning-Metric.csv')

    # 获取所有作物和模型名称
    crops = data['Crop'].unique()
    models_to_exclude = [
        'Freeze10-MTL(S2.5K+G1.25K)',
        'Freeze10-MTL(S2.5K+G2.5K)',
        'Freeze10-MTL(S2.5K+G5K)',
        'Freeze10-MTL(S2.5K+G10K)',
        'Freeze10-MTL(S2.5K+G20K)',
        'Freeze10(S2.5K)-DMT(S0.1K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S0.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S1.25K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G5K)'
    ]

    # 过滤掉不需要的模型
    data_filtered = data[~data['Model'].isin(models_to_exclude)]

    models_in_range1 = [
        'ChatGLM3_6B', 'Lora3', 'Lora5', 'Lora10', 'Lora15', 'Lora20', 'Lora30'
    ]
    models_in_range2 = [
        'ChatGLM3_6B', 'Freeze3', 'Freeze5', 'Freeze10', 'Freeze15', 'Freeze20', 'Freeze30'
    ]
    metrics_df_filtered1 = data_filtered[data_filtered['Model'].isin(models_in_range1)]
    metrics_df_filtered2 = data_filtered[data_filtered['Model'].isin(models_in_range2)]

    # 定义模型排序顺序
    model_order1 = [
        'ChatGLM3_6B', 'Lora3', 'Lora5', 'Lora10', 'Lora15', 'Lora20', 'Lora30'
    ]
    model_order2 = [
        'ChatGLM3_6B', 'Freeze3', 'Freeze5', 'Freeze10', 'Freeze15', 'Freeze20', 'Freeze30'
    ]

    # 设置字体属性
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 创建一个绘图，设置尺寸为（7, 8），并共享 Y 轴
    fig, axs = plt.subplots(2, 2, figsize=(7, 8), sharey=True)

    # 调整子图间距和边距，减小左右空白
    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.06, hspace=0.06)

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#7f4c7a', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#f7b6d2', '#c5b0d5', '#ffbb78', '#ff9896', '#c49c94'
    ]

    # 创建颜色字典
    color_dict = {crop: colors[i % len(colors)] for i, crop in enumerate(crops) if crop != 'average'}

    # 绘制每个作物的BLEU-4指数折线图
    for crop in crops:
        crop_data1 = metrics_df_filtered1[metrics_df_filtered1['Crop'] == crop]
        crop_data1['Model'] = pd.Categorical(crop_data1['Model'], categories=model_order1, ordered=True)
        crop_data1 = crop_data1.sort_values('Model')

        crop_data2 = metrics_df_filtered2[metrics_df_filtered2['Crop'] == crop]
        crop_data2['Model'] = pd.Categorical(crop_data2['Model'], categories=model_order2, ordered=True)
        crop_data2 = crop_data2.sort_values('Model')

        if crop != 'average':
            axs[0, 0].plot(crop_data1['Model'], crop_data1['BLEU-4(‰)'],
                        marker='o', label=crop.capitalize(), color=color_dict[crop])
            axs[0, 1].plot(crop_data2['Model'], crop_data2['BLEU-4(‰)'],
                        marker='o', label=crop.capitalize(), color=color_dict[crop])

    # 设置BLEU-4图表属性
    axs[0, 0].set_ylabel('BLEU-4 Index(‰)')
    axs[0, 0].grid()
    axs[0, 1].grid()

    # 绘制Average ROUGE F-score折线图
    for crop in crops:
        crop_data1 = metrics_df_filtered1[metrics_df_filtered1['Crop'] == crop]
        crop_data1['Model'] = pd.Categorical(crop_data1['Model'], categories=model_order1, ordered=True)
        crop_data1 = crop_data1.sort_values('Model')

        crop_data2 = metrics_df_filtered2[metrics_df_filtered2['Crop'] == crop]
        crop_data2['Model'] = pd.Categorical(crop_data2['Model'], categories=model_order2, ordered=True)
        crop_data2 = crop_data2.sort_values('Model')

        if crop != 'average':
            axs[1, 0].plot(crop_data1['Model'], crop_data1['Average ROUGE F-score(‰)'],
                        marker='o', label=crop.capitalize(), color=color_dict[crop])
            axs[1, 1].plot(crop_data2['Model'], crop_data2['Average ROUGE F-score(‰)'],
                        marker='o', label=crop.capitalize(), color=color_dict[crop])

    # 设置ROUGE图表属性
    axs[1, 0].set_ylabel('Average ROUGE F-score(‰)')
    axs[1, 0].grid()
    axs[1, 1].grid()

    # 去掉顶部子图的 x 轴标签
    axs[0, 0].set_xticklabels([])
    axs[0, 1].set_xticklabels([])

    model_order_display_ax1 = [
        'ChatGLM3-6B', 'LoRA3', 'LoRA5', 'LoRA10', 'LoRA15', 'LoRA20', 'LoRA30'
    ]
    model_order_display_ax2 = [
        'ChatGLM3-6B', 'Freeze3', 'Freeze5', 'Freeze10', 'Freeze15', 'Freeze20', 'Freeze30'
    ]
    # 设置 x 轴刻度
    axs[1, 0].set_xticks(range(len(model_order1)))
    axs[1, 0].set_xticklabels(model_order_display_ax1, rotation=45, ha='right')
    axs[1, 1].set_xticks(range(len(model_order2)))
    axs[1, 1].set_xticklabels(model_order_display_ax2, rotation=45, ha='right')

    # 添加子图标签
    axs[0, 0].text(0.55, 0.95, '(a)', transform=axs[0, 0].transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
    axs[0, 1].text(0.55, 0.95, '(b)', transform=axs[0, 1].transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
    axs[1, 0].text(0.55, 0.95, '(c)', transform=axs[1, 0].transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
    axs[1, 1].text(0.55, 0.95, '(d)', transform=axs[1, 1].transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')

    # 获取图例句柄和标签
    handles, labels = axs[0, 0].get_legend_handles_labels()
    labels = [label.capitalize() for label in labels]

    # 添加共享图例，放在顶部
    fig.legend(handles, labels, title='Crops', bbox_to_anchor=(0.52, 0.94),
            loc='center', ncol=5, fontsize=10)

    # 在保存图形时，使用 bbox_inches='tight' 去除多余空白
    plt.savefig('./fig/SFig1.png', dpi=500, bbox_inches='tight')
    plt.show()

def sfig2():
    data = pd.read_csv('./data/LLM/LLM_Metric/Fine-tuning-Metric/Fine-tuning-Metric.csv')

    # 获取所有作物和模型名称
    crops = data['Crop'].unique()
    models_to_exclude = [
        'ChatGLM3_6B','Lora3', 'Lora5', 'Lora10', 'Lora15', 'Lora20', 'Lora30',
        'Freeze3','Freeze5', 'Freeze15', 'Freeze20', 'Freeze30'
    ]

    # 过滤掉不需要的模型
    data_filtered = data[~data['Model'].isin(models_to_exclude)]

    models_in_range1 = [
        'Freeze10',
        'Freeze10-MTL(S2.5K+G1.25K)',
        'Freeze10-MTL(S2.5K+G2.5K)',
        'Freeze10-MTL(S2.5K+G5K)',
        'Freeze10-MTL(S2.5K+G10K)',
        'Freeze10-MTL(S2.5K+G20K)'
    ]
    models_in_range2 = [
        'Freeze10',
        'Freeze10(S2.5K)-DMT(S0.1K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S0.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S1.25K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G5K)'
    ]
    metrics_df_filtered1 = data_filtered[data_filtered['Model'].isin(models_in_range1)]
    metrics_df_filtered2 = data_filtered[data_filtered['Model'].isin(models_in_range2)]

    # 定义模型排序顺序
    model_order1 = [
        'Freeze10',
        'Freeze10-MTL(S2.5K+G1.25K)',
        'Freeze10-MTL(S2.5K+G2.5K)',
        'Freeze10-MTL(S2.5K+G5K)',
        'Freeze10-MTL(S2.5K+G10K)',
        'Freeze10-MTL(S2.5K+G20K)'
    ]
    model_order2 = [
        'Freeze10',
        'Freeze10(S2.5K)-DMT(S0.1K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S0.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S1.25K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G5K)'
    ]

    # 设置字体属性
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 增加图形宽度，创建子图
    fig, axs = plt.subplots(2, 2, figsize=(7, 8), sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整子图间距

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#7f4c7a', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#f7b6d2', '#c5b0d5', '#ffbb78', '#ff9896', '#c49c94'
    ]

    # 创建颜色字典
    color_dict = {crop: colors[i % len(colors)] for i, crop in enumerate(crops) if crop != 'average'}

    # 绘制每个作物的BLEU-4指数折线图
    for crop in crops:
        crop_data1 = metrics_df_filtered1[metrics_df_filtered1['Crop'] == crop]
        crop_data1['Model'] = pd.Categorical(crop_data1['Model'], categories=model_order1, ordered=True)
        crop_data1 = crop_data1.sort_values('Model')

        crop_data2 = metrics_df_filtered2[metrics_df_filtered2['Crop'] == crop]
        crop_data2['Model'] = pd.Categorical(crop_data2['Model'], categories=model_order2, ordered=True)
        crop_data2 = crop_data2.sort_values('Model')

        if crop != 'average':
            # 绘制 BLEU-4 指数，标签首字母大写
            axs[0, 0].plot(crop_data1['Model'], crop_data1['BLEU-4(‰)'],
                        marker='o', label=crop.capitalize(), color=color_dict[crop])
            axs[0, 1].plot(crop_data2['Model'], crop_data2['BLEU-4(‰)'],
                        marker='o', label=crop.capitalize(), color=color_dict[crop])

    # 设置BLEU-4图表属性
    axs[0, 0].set_ylabel('BLEU-4 Index(‰)')
    axs[0, 0].grid()
    axs[0, 1].grid()

    # 绘制Average ROUGE F-score折线图
    for crop in crops:
        crop_data1 = metrics_df_filtered1[metrics_df_filtered1['Crop'] == crop]
        crop_data1['Model'] = pd.Categorical(crop_data1['Model'], categories=model_order1, ordered=True)
        crop_data1 = crop_data1.sort_values('Model')

        crop_data2 = metrics_df_filtered2[metrics_df_filtered2['Crop'] == crop]
        crop_data2['Model'] = pd.Categorical(crop_data2['Model'], categories=model_order2, ordered=True)
        crop_data2 = crop_data2.sort_values('Model')

        if crop != 'average':
            # 绘制 ROUGE F-score，标签首字母大写
            axs[1, 0].plot(crop_data1['Model'], crop_data1['Average ROUGE F-score(‰)'],
                        marker='o', label=crop.capitalize(), color=color_dict[crop])
            axs[1, 1].plot(crop_data2['Model'], crop_data2['Average ROUGE F-score(‰)'],
                        marker='o', label=crop.capitalize(), color=color_dict[crop])

    # 设置ROUGE图表属性
    axs[1, 0].set_ylabel('Average ROUGE F-score(‰)')
    axs[1, 0].grid()
    axs[1, 1].grid()

    # 去掉顶部子图的 x 轴标签
    axs[0, 0].set_xticklabels([])
    axs[0, 1].set_xticklabels([])

    # 在标签中添加换行符，分成两行显示
    model_order_display_ax1 = [
        'Freeze10\n(S:2.5K)',
        'Freeze10-MTL\n(S:2.5K+G:1.25K)',
        'Freeze10-MTL\n(S:2.5K+G:2.5K)',
        'Freeze10-MTL\n(S:2.5K+G:5K)',
        'Freeze10-MTL\n(S:2.5K+G:10K)',
        'Freeze10-MTL\n(S:2.5K+G:20K)'
    ]
    model_order_display_ax2 = [
        'Freeze10\n(S:2.5K)',
        'Freeze10(S:2.5K)-DMT\n(S:0.1K+G:2.5K)',
        'Freeze10(S:2.5K)-DMT\n(S:0.5K+G:2.5K)',
        'Freeze10(S:2.5K)-DMT\n(S:1.25K+G:2.5K)',
        'Freeze10(S:2.5K)-DMT\n(S:2.5K+G:2.5K)',
        'Freeze10(S:2.5K)-DMT\n(S:2.5K+G:5K)'
    ]

    # 设置 x 轴刻度和标签，旋转并右对齐
    axs[1, 0].set_xticks(range(len(model_order1)))
    axs[1, 0].set_xticklabels(model_order_display_ax1, rotation=90)
    axs[1, 1].set_xticks(range(len(model_order2)))
    axs[1, 1].set_xticklabels(model_order_display_ax2, rotation=90)

    # 添加子图标签
    axs[0, 0].text(0.46, 0.95, '(a)', transform=axs[0, 0].transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')
    axs[0, 1].text(0.455, 0.95, '(b)', transform=axs[0, 1].transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')
    axs[1, 0].text(0.46, 0.95, '(c)', transform=axs[1, 0].transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')
    axs[1, 1].text(0.46, 0.95, '(d)', transform=axs[1, 1].transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    # 获取图例句柄和标签
    handles, labels = axs[0, 0].get_legend_handles_labels()
    # 将图例标签首字母大写
    labels = [label.capitalize() for label in labels]
    # 添加共享图例，放在顶部
    fig.legend(handles, labels, title='Crops', bbox_to_anchor=(0.53, 0.94),
            loc='center', ncol=5, fontsize=10)

    # 调整整体布局，防止标签和图例被截断
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig('./fig/SFig2.png', dpi=500)
    plt.show()

def sfig3():
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 读取第一个CSV文件
    df_cmmlu = pd.read_csv('./data/LLM/LLM_Metric/CMMLU-Metric/CMMLU_Metric.csv')

    # 过滤出 Subject 等于 'Overall'
    df_cmmlu = df_cmmlu[df_cmmlu['Subject'] != 'Overall']

    # 去除指定的 Model
    models_to_remove = [
        'Freeze10-MTL(S2.5K+G1.25K)',
        'Freeze10-MTL(S2.5K+G2.5K)',
        'Freeze10-MTL(S2.5K+G5K)',
        'Freeze10-MTL(S2.5K+G10K)',
        'Freeze10-MTL(S2.5K+G20K)',
        'Freeze10(S2.5K)-DMT(S0.1K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S0.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S1.25K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G5K)'
    ]
    df_cmmlu = df_cmmlu[~df_cmmlu['Model'].isin(models_to_remove)]

    # 原始模型名称与新模型名称的映射
    model_name_mapping = {
        'ChatGLM3_6B': 'ChatGLM3-6B',
        'Lora3': 'LoRA3',
        'Lora5': 'LoRA5',
        'Lora10': 'LoRA10',
        'Lora15': 'LoRA15',
        'Lora20': 'LoRA20',
        'Lora30': 'LoRA30',
        'Freeze3': 'Freeze3',
        'Freeze5': 'Freeze5',
        'Freeze10': 'Freeze10',
        'Freeze15': 'Freeze15',
        'Freeze20': 'Freeze20',
        'Freeze30': 'Freeze30'
    }

    # 替换模型名称
    df_cmmlu['Model'] = df_cmmlu['Model'].replace(model_name_mapping)
    
    model_order =['ChatGLM3-6B', 'LoRA3', 'LoRA5', 'LoRA10', 'LoRA15', 'LoRA20', 'LoRA30',
        'Freeze3', 'Freeze5', 'Freeze10', 'Freeze15', 'Freeze20', 'Freeze30']

    df_cmmlu['Model'] = pd.Categorical(df_cmmlu['Model'], categories=model_order, ordered=True)

    # 指定类别顺序
    subject_order = ['STEM', 'Humanities', 'Social Science', 'Other', 'China specific']
    df_cmmlu['Subject'] = pd.Categorical(df_cmmlu['Subject'], categories=subject_order, ordered=True)
    df_cmmlu = df_cmmlu.sort_values('Subject')

    # 创建一个包含两个子图的图形
    fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # a子图：0-shot
    ax1 = sns.barplot(data=df_cmmlu[df_cmmlu['Shot'] == 0], x='Subject', y='Accuracy', hue='Model', 
                      palette='Set2', ax=axs[0], zorder=2)
    ax1.set_ylim((40, 60))
    ax1.set_ylabel('CMMLU Accuracy (%)')
    ax1.set_title('(a) Zero-shot', fontsize=14, fontweight='bold', y=0.90)
    ax1.grid(True, zorder=1)

    # b子图：5-shot
    ax2 = sns.barplot(data=df_cmmlu[df_cmmlu['Shot'] == 5], x='Subject', y='Accuracy', hue='Model', 
                      palette='Set2', ax=axs[1], zorder=2)
    ax2.set_ylim((35, 58))
    ax2.set_ylabel('CMMLU Accuracy (%)')
    ax2.set_title('(b) Five-shot', fontsize=14, fontweight='bold', y=0.90)
    ax2.grid(True, zorder=1)

    # 修改图例文字，保持颜色不变
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=5, columnspacing=1.8)
    ax2.legend_.remove()

    plt.tight_layout()
    plt.savefig('./fig/SFig3.png', dpi=500, bbox_inches='tight')
    plt.show()

def sfig4():
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 读取第一个CSV文件
    df_cmmlu = pd.read_csv('./data/LLM/LLM_Metric/CMMLU-Metric/CMMLU_Metric.csv')

    # 过滤出 Subject 等于 'Overall'
    df_cmmlu = df_cmmlu[df_cmmlu['Subject'] != 'Overall']

    # 去除指定的 Model
    models_to_remove = [
        'ChatGLM3_6B','Lora3', 'Lora5', 'Lora10', 'Lora15', 'Lora20', 'Lora30',
        'Freeze3','Freeze5', 'Freeze15', 'Freeze20', 'Freeze30'
    ]
    df_cmmlu = df_cmmlu[~df_cmmlu['Model'].isin(models_to_remove)]


    # 原始模型名称与新模型名称的映射
    model_name_mapping = {
        'Freeze10': 'Freeze10(S:2.5K)',
        'Freeze10-MTL(S2.5K+G1.25K)': 'Freeze10-MTL(S:2.5K+G:1.25K)',
        'Freeze10-MTL(S2.5K+G2.5K)': 'Freeze10-MTL(S:2.5K+G:2.5K)',
        'Freeze10-MTL(S2.5K+G5K)': 'Freeze10-MTL(S:2.5K+G:5K)',
        'Freeze10-MTL(S2.5K+G10K)': 'Freeze10-MTL(S:2.5K+G:10K)',
        'Freeze10-MTL(S2.5K+G20K)': 'Freeze10-MTL(S:2.5K+G:20K)',
        'Freeze10(S2.5K)-DMT(S0.1K+G2.5K)': 'Freeze10(S:2.5K)-DMT(S:0.1K+G:2.5K)',
        'Freeze10(S2.5K)-DMT(S0.5K+G2.5K)': 'Freeze10(S:2.5K)-DMT(S:0.5K+G:2.5K)',
        'Freeze10(S2.5K)-DMT(S1.25K+G2.5K)': 'Freeze10(S:2.5K)-DMT(S:1.25K+G:2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G2.5K)': 'Freeze10(S:2.5K)-DMT(S:2.5K+G:2.5K)',
        'Freeze10(S2.5K)-DMT(S2.5K+G5K)': 'Freeze10(S:2.5K)-DMT(S:2.5K+G:5K)'
    }

        # 替换模型名称
    df_cmmlu['Model'] = df_cmmlu['Model'].replace(model_name_mapping)

    # 指定模型顺序
    # model_order = ['freeze10','freeze10_1.25K_alpaca','freeze10_2.5K_alpaca', 'freeze10_5K_alpaca', 'freeze10_10K_alpaca', 'freeze10_20K_alpaca',
    #   'freeze10_DMT(1-1`25)','freeze10_DMT(1-1`5)','freeze10_DMT(1-1`2)','freeze10_DMT(1-1)','freeze10_DMT(2-1)']
    model_order = ['Freeze10(S:2.5K)', 'Freeze10-MTL(S:2.5K+G:1.25K)','Freeze10-MTL(S:2.5K+G:2.5K)', 'Freeze10-MTL(S:2.5K+G:5K)', 'Freeze10-MTL(S:2.5K+G:10K)', 'Freeze10-MTL(S:2.5K+G:20K)',
     'Freeze10(S:2.5K)-DMT(S:0.1K+G:2.5K)', 'Freeze10(S:2.5K)-DMT(S:0.5K+G:2.5K)', 'Freeze10(S:2.5K)-DMT(S:1.25K+G:2.5K)', 'Freeze10(S:2.5K)-DMT(S:2.5K+G:2.5K)','Freeze10(S:2.5K)-DMT(S:2.5K+G:5K)']
    df_cmmlu['Model'] = pd.Categorical(df_cmmlu['Model'], categories=model_order, ordered=True)

    # 指定类别顺序
    subject_order = ['STEM', 'Humanities', 'Social Science', 'Other', 'China specific']
    df_cmmlu['Subject'] = pd.Categorical(df_cmmlu['Subject'], categories=subject_order, ordered=True)
    df_cmmlu = df_cmmlu.sort_values('Subject')

    # 创建一个包含两个子图的图形
    fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # a子图：0-shot
    ax1 = sns.barplot(data=df_cmmlu[df_cmmlu['Shot'] == 0], x='Subject', y='Accuracy', hue='Model', 
                      palette='Set2', ax=axs[0], zorder=2)
    ax1.set_ylim((40, 60))
    ax1.set_ylabel('CMMLU Accuracy (%)')
    ax1.set_title('(a) Zero-shot', fontsize=14, fontweight='bold', y=0.90)
    ax1.grid(True, zorder=1)

    # b子图：5-shot
    ax2 = sns.barplot(data=df_cmmlu[df_cmmlu['Shot'] == 5], x='Subject', y='Accuracy', hue='Model', 
                      palette='Set2', ax=axs[1], zorder=2)
    ax2.set_ylim((35, 58))
    ax2.set_ylabel('CMMLU Accuracy (%)')
    ax2.set_title('(b) Five-shot', fontsize=14, fontweight='bold', y=0.90)
    ax2.grid(True, zorder=1)

    # 修改图例文字，保持颜色不变
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.48), ncol=2, columnspacing=5.0)
    ax2.legend_.remove()

    plt.tight_layout()
    plt.savefig('./fig/SFig4.png', dpi=500, bbox_inches='tight')
    plt.show()

def sfig5():
    # 设置全局字体大小和字体样式
    plt.rcParams.update({'font.size': 10, 'font.family': 'Times New Roman'})

    # 加载 CSV 文件
    data = pd.read_csv('./data/CV/CV_Metric/CV_confusion_matrix.csv', index_col=0)

    # 初始化分类错误计数的字典
    misclassifications = {}

    # 初始化特定分类错误的计数器
    healthy_to_disease_count = 0
    disease_to_healthy_count = 0

    # 遍历混淆矩阵
    for actual_label in data.index:
        for predicted_label in data.columns:
            if actual_label != predicted_label:  # 仅考虑错误分类
                count = data.loc[actual_label, predicted_label]
                if count > 0:  # 仅包含非零错误分类
                    if 'healthy' in actual_label and 'healthy' not in predicted_label:
                        healthy_to_disease_count += count
                        misclassification_pair = f"{actual_label} → {predicted_label}"
                        misclassifications[misclassification_pair] = (count, 'healthy-disease')
                    elif 'healthy' not in actual_label and 'healthy' in predicted_label:
                        disease_to_healthy_count += count
                        misclassification_pair = f"{actual_label} → {predicted_label}"
                        misclassifications[misclassification_pair] = (count, 'disease-healthy')
                    else:
                        misclassification_pair = f"{actual_label} → {predicted_label}"
                        if 'healthy' in actual_label and 'healthy' in predicted_label:
                            category = 'healthy-healthy'
                        else:
                            category = 'disease-disease'
                        misclassifications[misclassification_pair] = (count, category)

    # 输出特定分类错误的总计数
    print(f"Total 'healthy predicted as disease' count: {healthy_to_disease_count}")
    print(f"Total 'disease predicted as healthy' count: {disease_to_healthy_count}")

    # 将 "healthy - disease" 和 "disease - healthy" 的总计数添加到字典
    misclassifications["Healthy → Disease"] = (healthy_to_disease_count, 'healthy-disease')
    misclassifications["Disease → Healthy"] = (disease_to_healthy_count, 'disease-healthy')

    # 将字典转换为 DataFrame 并按计数降序排序，选择前40个
    misclassifications_df = pd.DataFrame(
        [(pair, count, category) for pair, (count, category) in misclassifications.items()],
        columns=["Misclassification Pair", "Count", "Category"]
    )
    misclassifications_df = misclassifications_df.sort_values(by="Count", ascending=False).head(40).reset_index(drop=True)

    # 添加省略行以表示继续
    ellipsis_row = pd.DataFrame([{"Misclassification Pair": "", "Count": None, "Category": None}])
    misclassifications_df = pd.concat([misclassifications_df, ellipsis_row], ignore_index=True)

    # 定义 Seaborn 颜色
    palette = sns.color_palette("Set2", 4)
    color_map = {
        'healthy-healthy': palette[0],
        'disease-disease': palette[1],
        'healthy-disease': palette[2],
        'disease-healthy': palette[3]
    }
    colors = [color_map[category] if category in color_map else 'gray' for category in misclassifications_df['Category']]

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(3.5, min(0.4 * len(misclassifications_df), 8)))

    # 排除省略行的值
    ax.barh(misclassifications_df["Misclassification Pair"][:-1], misclassifications_df["Count"][:-1], color=colors[:-1])
    ax.set_xlabel("Count")
    ax.invert_yaxis()

    # 设置 x 轴刻度从 0 到 80，间隔为 10
    ax.set_xticks(range(0, 81, 10))

    # 在 y 轴标签上方添加 "True → Predict"
    ax.annotate("True → Predict", xy=(-0.26, 0.97), xycoords="axes fraction", ha="center", fontsize=14, fontweight="bold")
    ax.annotate("···", xy=(-0.08, 0.0155), xycoords="axes fraction", ha="center", fontsize=18, fontweight="bold")

    # 调整 y 轴刻度以显示省略号
    ax.set_yticks(range(len(misclassifications_df)))
    ax.set_yticklabels(misclassifications_df["Misclassification Pair"])

    # 创建自定义图例，并将方块缩短
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=palette[0], lw=4, label='Healthy → Healthy', markersize=7),
        Line2D([0], [0], color=palette[1], lw=4, label='Disease → Disease', markersize=7),
        Line2D([0], [0], color=palette[2], lw=4, label='Healthy → Disease', markersize=7),
        Line2D([0], [0], color=palette[3], lw=4, label='Disease → Healthy', markersize=7)
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 1.09), ncol=2, prop={'family': 'Times New Roman', 'size': 10})

    # 保存图片
    output_image_path = './fig/SFig5.png'
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def sfig6():
    plt.rcParams['font.family'] = 'Times New Roman'

    # 加载 CSV 文件
    data = pd.read_csv('./data/CV/CV_Metric/CV_confusion_matrix.csv', index_col=0)

    # 提取所有标签
    labels = data.index.to_list()

    # 提取作物和病害信息的函数
    def extract_crop_disease(label):
        parts = label.split()
        crop = parts[0]
        disease = ' '.join(parts[1:])
        return crop, disease

    # 按作物和病害分组标签
    crop_dict = {}
    disease_dict = {}

    for label in labels:
        crop, disease = extract_crop_disease(label)
        if crop not in crop_dict:
            crop_dict[crop] = []
        if disease not in disease_dict:
            disease_dict[disease] = []
        crop_dict[crop].append(label)
        disease_dict[disease].append(label)

    # 计算整体作物级别的准确率（同一作物内的预测视为正确）
    correct_crop_predictions = 0
    total_predictions = data.values.sum()

    for crop, crop_labels in crop_dict.items():
        crop_indices = [labels.index(label) for label in crop_labels]
        crop_matrix = data.values[np.ix_(crop_indices, crop_indices)]
        correct_crop_predictions += crop_matrix.sum()

    overall_crop_accuracy = (correct_crop_predictions / total_predictions) * 100  # 转换为百分比

    # 计算每个作物和每种病害的准确率
    crop_accuracies = {}
    for crop, crop_labels in crop_dict.items():
        matrix = data.loc[crop_labels, crop_labels].values
        accuracy = np.trace(matrix) / np.sum(matrix)  # 计算准确率
        crop_accuracies[crop] = accuracy * 100  # 转换为百分比

    disease_accuracies = {}
    for disease, disease_labels in disease_dict.items():
        if len(disease_labels) > 1:  # 只考虑有多个标签的病害
            matrix = data.loc[disease_labels, disease_labels].values
            accuracy = np.trace(matrix) / np.sum(matrix)  # 计算准确率
            disease_accuracies[disease] = accuracy * 100  # 转换为百分比

    # 提取所有预测和真实标签
    y_true = []
    y_pred = []
    for i, label_row in enumerate(labels):
        for j, label_col in enumerate(labels):
            y_true.extend([i] * int(data.values[i, j]))
            y_pred.extend([j] * int(data.values[i, j]))

    # 计算整体 F1、Recall 和 Precision
    overall_precision = precision_score(y_true, y_pred, average='macro') * 100
    overall_recall = recall_score(y_true, y_pred, average='macro') * 100
    overall_f1 = f1_score(y_true, y_pred, average='macro') * 100

    # 从 Seaborn 调色板中选择颜色
    color_accuracy = sns.color_palette("Set2", n_colors=2)[0]  # 用于 Accuracy 和 第二、第三个图
    color_precision = sns.color_palette("Set2", n_colors=3)[1]  # Precision 的颜色
    color_recall = sns.color_palette("Set2", n_colors=3)[2]    # Recall 的颜色
    color_f1 = sns.color_palette("Set2", n_colors=4)[3]        # F1 Score 的颜色

    # 在一个图中绘制所有图表
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [0.5, 1, 1]})
    sns.set(style='whitegrid')

    # 设置坐标轴框架的 zorder 为最高
    for ax in [ax0, ax1, ax2]:
        ax.spines['top'].set_zorder(3)
        ax.spines['right'].set_zorder(3)
        ax.spines['bottom'].set_zorder(3)
        ax.spines['left'].set_zorder(3)

    # 整体指标柱状图，使用不同颜色并去掉边缘
    ax0.bar(['Accuracy'], [overall_crop_accuracy], color=color_accuracy, edgecolor=color_accuracy, width=0.4, zorder=2, label="Accuracy")
    ax0.bar(['Precision'], [overall_precision], color=color_precision, edgecolor=color_precision, width=0.4, zorder=2, label="Precision")
    ax0.bar(['Recall'], [overall_recall], color=color_recall, edgecolor=color_recall, width=0.4, zorder=2, label="Recall")
    ax0.bar(['F1 Score'], [overall_f1], color=color_f1, edgecolor=color_f1, width=0.4, zorder=2, label="F1 Score")
    ax0.set_ylabel('Accuracy (%)')
    ax0.set_ylim(90, 103)
    ax0.set_yticks(range(90, 104, 5))  # 固定 y 轴刻度间隔为 5
    ax0.grid(axis='y', linestyle='-', zorder=1)
    for i, v in enumerate([overall_crop_accuracy, overall_precision, overall_recall, overall_f1]):
        ax0.text(i, v + 0.3, f"{v:.1f}", ha='center', fontsize=10, zorder=4, fontname='Times New Roman')
    ax0.text(0.5, 0.8, '(a)', fontname='Times New Roman', transform=ax0.transAxes, fontsize=14, fontweight='bold', ha='center')

    # 作物准确率的柱状图，使用 color_accuracy 颜色并去掉边缘
    ax1.bar(crop_accuracies.keys(), crop_accuracies.values(), color=color_accuracy, edgecolor=color_accuracy, width=0.4, zorder=2)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(90, 102)
    ax1.set_yticks(range(90, 101, 2))  # 固定 y 轴刻度间隔为 5
    ax1.grid(axis='y', linestyle='-', zorder=1)
    ax1.set_xticks(range(len(crop_accuracies)))  # 设置 x 轴的刻度
    ax1.set_xticklabels(crop_accuracies.keys(), rotation=45, ha='right', fontsize=10)
    for i, v in enumerate(crop_accuracies.values()):
        ax1.text(i, v + 0.3, f"{v:.1f}", ha='center', fontsize=10, zorder=4, fontname='Times New Roman')
    ax1.text(0.5, 0.9, '(b)', fontname='Times New Roman', transform=ax1.transAxes, fontsize=14, fontweight='bold', ha='center')

    # 病害准确率的柱状图，使用 color_accuracy 颜色并去掉边缘
    disease_labels = [disease.capitalize() for disease in disease_accuracies.keys()]
    ax2.bar(disease_labels, disease_accuracies.values(), color=color_accuracy, edgecolor=color_accuracy, width=0.4, zorder=2)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(90, 102)
    ax2.set_yticks(range(90, 101, 2))  # 固定 y 轴刻度间隔为 5
    ax2.grid(axis='y', linestyle='-', zorder=1)
    ax2.set_xticks(range(len(disease_accuracies)))  # 设置 x 轴的刻度
    ax2.set_xticklabels(disease_labels, rotation=45, ha='right', fontsize=10)
    for i, v in enumerate(disease_accuracies.values()):
        ax2.text(i, v + 0.3, f"{v:.1f}", ha='center', fontsize=10, zorder=4, fontname='Times New Roman')
    ax2.text(0.5, 0.9, '(c)', fontname='Times New Roman', transform=ax2.transAxes, fontsize=14, fontweight='bold', ha='center')

    # 添加图例在顶部
    handles = [
        plt.Line2D([0], [0], color=color_accuracy, lw=4, label='Accuracy'),
        plt.Line2D([0], [0], color=color_precision, lw=4, label='Precision'),
        plt.Line2D([0], [0], color=color_recall, lw=4, label='Recall'),
        plt.Line2D([0], [0], color=color_f1, lw=4, label='F1 Score')
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.89), ncol=4, title='', prop={'family': 'Times New Roman', 'size': 10})

    # 调整整体间距和只缩短 a-b 之间的间距
    fig.subplots_adjust(bottom=0.1, wspace=0.2, hspace=0.4)  # 设置默认hspace
    ax0.set_position([ax0.get_position().x0, ax0.get_position().y0 - 0.04, ax0.get_position().width, ax0.get_position().height])  # 上移 ax0

    output_combined_path = r'fig/SFig6.png'
    plt.savefig(output_combined_path, dpi=500, bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    # fig5()
    fig7()
    # fig8()
    # fig9()
    # fig10()
    # fig11()
    # sfig1()
    # sfig2()
    # sfig3()
    # sfig4()
    # sfig5()
    # sfig6()
