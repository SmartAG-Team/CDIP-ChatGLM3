import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def fig7():
    labels = []
    sizes = []
    directory = './data/LLM_fine-tuning_dataset/summary'
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            varieties = file
            name = varieties.replace('.json', '')
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

    # plt.figure(figsize=(7, 4))  # 设置图形大小
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
    plt.savefig('./fig/Fig.7.png', dpi=300)
    plt.show()

def fig8():
    models = [
        'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
        'ViT_bp16', 'ViT_bp16_in21k', 'ViT_bp32', 'ViT_bp32_in21k',
        'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161',
        'EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'EfficientNet-B3'
    ]

    # Performance metrics data (updated)
    F1 = [0.961, 0.957, 0.963, 0.962, 0.949, 0.899, 0.934, 0.934, 0.955, 0.958, 0.963, 0.964, 0.967, 0.967, 0.969, 0.965]
    Recall = [0.960, 0.956, 0.962, 0.962, 0.950, 0.894, 0.936, 0.936, 0.958, 0.957, 0.964, 0.965, 0.967, 0.967, 0.969, 0.965]
    Accuracy = [0.970, 0.968, 0.970, 0.969, 0.957, 0.927, 0.946, 0.946, 0.965, 0.969, 0.970, 0.971, 0.972, 0.973, 0.975, 0.971]
    Precision = [0.967, 0.965, 0.965, 0.964, 0.949, 0.917, 0.938, 0.938, 0.954, 0.964, 0.964, 0.966, 0.968, 0.968, 0.970, 0.965]

    # Bar width
    bar_width = 0.5
    index = np.arange(len(models))

    # Create subplots with shared x-axis
    fig, axs = plt.subplots(2, 2, figsize=(10, 12), sharex=True)

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Plot each metric in a separate subplot
    metrics = [F1, Recall, Accuracy, Precision]
    metric_names = ['F1', 'Recall', 'Accuracy', 'Precision']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # high contrast colors

    # Parameters for adjusting title positions
    title_x_position = 0.4 # x position of the title (0.0 to 1.0)
    title_pad = 7 # padding between the title and the plot

    for i, (ax, metric, name, color) in enumerate(zip(axs, metrics, metric_names, colors)):
        bars = ax.barh(index, metric, bar_width, label=name, color=color, edgecolor=color)

        # Set labels and title with adjustable position
        if i % 2 == 0:  # Only set y-tick labels for the left column
            ax.set_yticks(index)
            ax.set_yticklabels(models, fontname='Times New Roman', fontsize=10)
        else:  # Remove y-tick labels for the right column
            ax.set_yticks(index)
            ax.set_yticklabels([])

        # Annotate the highest value with a star and the value
        max_value = max(metric)
        max_indices = [j for j, val in enumerate(metric) if val == max_value]

        for max_index in max_indices:
            ax.annotate(f'★ {max_value}', xy=(max_value, max_index), xytext=(max_value + 0.01, max_index - 0.21),
                        textcoords='data', fontsize=8, color='black', ha='center')

        # Custom legend with star for highest value
        custom_label = f'{name} (highest: ★)'
        ax.legend([custom_label], loc='center left', bbox_to_anchor=(0.3, 1.04), prop={'size': 10})

        # Set x-axis range for each metric
        if name == 'F1' or name == 'Precision':
            ax.set_xlim(0.85, 1.00)
        else:
            ax.set_xlim(0.82, 1.00)

    # Set x-axis label for the bottom subplots
    for ax in axs[-2:]:
        ax.set_xlabel('Percentage(%)', fontname='Times New Roman', fontsize=14)

    # Adjust layout and reduce horizontal space between subplots
    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()

    # Save the figure with higher resolution
    plt.savefig('./fig/Fig.8.png', dpi=300)
    plt.show()

def fig9():
    # 读取CSV文件
    data = pd.read_csv('D:\student\lzy\CDIP-ChatGLM3\data\LLM_Metric\sum_metrics.csv')

    # 获取所有作物和模型名称
    crops = data['Crop'].unique()
    models_to_exclude = ['freeze10_alpaca_10K', 'freeze10_alpaca_20K', 'freeze10_alpaca_5K']

    # 过滤掉不需要的模型
    data_filtered = data[~data['Model'].isin(models_to_exclude)]

    # 定义模型排序顺序
    model_order = ['chatglm6b', 'lora3', 'lora5', 'lora10', 'freeze3', 'freeze5', 'freeze10']

    # 设置字体属性
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 创建一个绘图，设置尺寸为（7, 10）
    fig, axs = plt.subplots(2, 1, figsize=(7, 8))

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#7f4c7a', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#f7b6d2', '#c5b0d5', '#ffbb78', '#ff9896', '#c49c94'
    ]

    # 创建颜色字典
    color_dict = {crop: colors[i % len(colors)] for i, crop in enumerate(crops) if crop != 'average'}
    color_dict['average'] = '#1f77b4'  # 将红色分配给average类别

    # 绘制每个作物的BLEU-4指数折线图
    for crop in crops:
        crop_data = data_filtered[data_filtered['Crop'] == crop]
        
        # 将模型按照定义的顺序排序
        crop_data['Model'] = pd.Categorical(crop_data['Model'], categories=model_order, ordered=True)
        crop_data = crop_data.sort_values('Model')
        
        if crop == 'average':
            axs[0].plot(crop_data['Model'], crop_data['BLEU-4(‰)'], marker='o', label=crop, color=color_dict[crop], linewidth=2)
            for i in range(len(crop_data)):
                model_name = crop_data['Model'].iloc[i]
                score_value = crop_data['BLEU-4(‰)'].iloc[i]
                if i == 0:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(12, 5), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 1:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(5, 10), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 2:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(5, 5), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 3:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(5, 5), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 4:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(5, 10), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 5:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(12, 0), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 6:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(-8, 8), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
        # else:    
        #     axs[0].plot(crop_data['Model'], crop_data['BLEU-4(‰)'], marker='o', label=crop, color=color_dict[crop])

    # 设置BLEU-4图表属性
    axs[0].set_ylabel('BLEU-4 Index(‰)')
    axs[0].set_xticklabels([])
    axs[0].grid(alpha=0.5)
    axs[0].set_ylim(50, 120)  # 设置y轴上限为100
    # 绘制Average ROUGE F-score折线图
    for crop in crops:
        crop_data = data_filtered[data_filtered['Crop'] == crop]
        
        # 将模型按照定义的顺序排序
        crop_data['Model'] = pd.Categorical(crop_data['Model'], categories=model_order, ordered=True)
        crop_data = crop_data.sort_values('Model')

        if crop == 'average':
            axs[1].plot(crop_data['Model'], crop_data['Average ROUGE F-score(‰)'], marker='o', label=crop, color=color_dict[crop], linewidth=2)
            for i in range(len(crop_data)):
                model_name = crop_data['Model'].iloc[i]
                score_value = crop_data['Average ROUGE F-score(‰)'].iloc[i]
                if i == 0:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(12, 0), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 1:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(5, 10), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 2:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(5, 5), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 3:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(5, 5), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 4:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(0, 10), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 5:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(12, 0), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 6:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(-10, 8), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
        # else:
        #     axs[1].plot(crop_data['Model'], crop_data['Average ROUGE F-score(‰)'], marker='o', label=crop, color=color_dict[crop])

    # 设置ROUGE图表属性
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Average ROUGE F-score(‰)')
    axs[1].set_xticklabels(crop_data['Model'])
    axs[1].grid(alpha=0.5)
    axs[1].set_ylim(65, 95)  # 设置y轴上限为100
    for tick in axs[0].get_yticklabels():
        tick.set_fontname('Times New Roman')
    for tick in axs[1].get_yticklabels():
        tick.set_fontname('Times New Roman')
    # 保存并显示图表
    plt.tight_layout()
    plt.savefig('./fig/Fig.9.png', dpi=300)

def fig10():
    data = pd.read_csv('D:\student\lzy\CDIP-ChatGLM3\data\LLM_Metric\sum_metrics.csv')

    # 获取所有作物和模型名称
    crops = data['Crop'].unique()
    models_to_exclude = ['chatglm6b', 'freeze3', 'freeze5','lora5','lora10','lora3']

    # 过滤掉不需要的模型
    data_filtered = data[~data['Model'].isin(models_to_exclude)]

    # 定义模型排序顺序
    model_order = ['freeze10', 'freeze10_alpaca_5K', 'freeze10_alpaca_10K', 'freeze10_alpaca_20K']

    # 设置字体属性
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 创建一个绘图，设置尺寸为（7, 8）
    fig, axs = plt.subplots(2, 1, figsize=(7, 8))

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#7f4c7a', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#f7b6d2', '#c5b0d5', '#ffbb78', '#ff9896', '#c49c94'
    ]

    # 创建颜色字典
    color_dict = {crop: colors[i % len(colors)] for i, crop in enumerate(crops) if crop != 'average'}
    color_dict['average'] = '#1f77b4'  # 将红色分配给average类别

    for crop in crops:
        crop_data = data_filtered[data_filtered['Crop'] == crop]
        
        # 将模型按照定义的顺序排序
        crop_data['Model'] = pd.Categorical(crop_data['Model'], categories=model_order, ordered=True)
        crop_data = crop_data.sort_values('Model')
        
        if crop == 'average':
            axs[0].plot(crop_data['Model'], crop_data['BLEU-4(‰)'], marker='o', label=crop, color=color_dict[crop], linewidth=2)
            for i in range(len(crop_data)):
                model_name = crop_data['Model'].iloc[i]
                score_value = crop_data['BLEU-4(‰)'].iloc[i]
                if i == 0:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(12, 5), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 1:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(0, 15), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 2:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(0, 5), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 3:
                    axs[0].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(-8, 5), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
        # else:    
        #     axs[0].plot(crop_data['Model'], crop_data['BLEU-4(‰)'], marker='o', label=crop, color=color_dict[crop])

    # 设置BLEU-4图表属性
    axs[0].set_ylabel('BLEU-4 Index(‰)')
    axs[0].set_xticklabels([])
    axs[0].set_ylim(105, 130)  # 设置y轴上限为100
    # axs[0].legend(title='Crops', bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=5)
    axs[0].grid()
    axs[0].grid(alpha=0.5)  # 设置网格线的透明度
    # 绘制Average ROUGE F-score折线图
    for crop in crops:
        crop_data = data_filtered[data_filtered['Crop'] == crop]
        
        # 将模型按照定义的顺序排序
        crop_data['Model'] = pd.Categorical(crop_data['Model'], categories=model_order, ordered=True)
        crop_data = crop_data.sort_values('Model')

        if crop == 'average':
            axs[1].plot(crop_data['Model'], crop_data['Average ROUGE F-score(‰)'], marker='o', label=crop, color=color_dict[crop], linewidth=2)

            # 遍历所有模型，标注每个点的值
            for i in range(len(crop_data)):
                model_name = crop_data['Model'].iloc[i]
                score_value = crop_data['Average ROUGE F-score(‰)'].iloc[i]
                if i == 0:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(12, 5), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 1:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(5, 10), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 2:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(5, 5), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')
                if i == 3:
                    axs[1].annotate(f'{score_value:.2f}', 
                                    xy=(model_name, score_value),
                                    xytext=(-5, 8), 
                                    textcoords='offset points',
                                    fontsize=10,
                                    color='#000000')

        # else:
        #     axs[1].plot(crop_data['Model'], crop_data['Average ROUGE F-score(‰)'], marker='o', label=crop, color=color_dict[crop])

    # 设置ROUGE图表属性
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Average ROUGE F-score(‰)')
    axs[1].set_ylim(90, 100)  # 设置y轴上限为100
    axs[1].set_xticklabels(crop_data['Model'])
    axs[1].grid(alpha=0.5)  # 设置网格线的透明度
    for tick in axs[0].get_yticklabels():
        tick.set_fontname('Times New Roman')
    for tick in axs[1].get_yticklabels():
        tick.set_fontname('Times New Roman')
    # 保存并显示图表
    plt.tight_layout()
    plt.savefig('./fig/Fig.10.png', dpi=300)

def fig11():
    data = pd.read_csv(r'D:\student\lzy\CDIP-ChatGLM3\data\LLM_CMMLU\accuracy\summary\results_0shot.csv')
    data_2 = pd.read_csv(r'D:\student\lzy\CDIP-ChatGLM3\data\LLM_CMMLU\accuracy\summary\results_0shot.csv')  # 确保这是不同的数据文件
    # 指定模型的摆放顺序
    model_order = ['chatglm3_6b', 'freeze10', 'freeze10_alpaca_5K', 'freeze10_alpaca_10K', 'freeze10_alpaca_20K']
    data['Model'] = pd.Categorical(data['Model'], categories=model_order, ordered=True)
    data_2['Model'] = pd.Categorical(data_2['Model'], categories=model_order, ordered=True)  # 确保第二个数据的模型顺序相同

    # 获取每个Subject的模型和准确性
    subjects = data['Subject'].unique()
    models = data['Model'].cat.categories

    # 设置柱状图的宽度
    bar_width = 0.15
    x = np.arange(len(subjects))

    # 创建绘图
    fig, axs = plt.subplots(2, 1, figsize=(7, 8))
    axs[0].grid(alpha=0.5, zorder=0)  # 设置网格线的 zorder
    axs[1].grid(alpha=0.5, zorder=0)

    colors = ['#6C96CC', '#96D2B0', '#F1C89A', '#D7DDDF', '#ED6E69']

    # 第一个子图
    for i, model in enumerate(models):
        model_data = data[data['Model'] == model]['Accuracy']
        axs[0].bar(x + i * bar_width, model_data, width=bar_width, label=model, color=colors[i], zorder=2)  # 设置柱状图的 zorder

    axs[0].set_ylabel('Accuracy', fontsize=10, fontname='Times New Roman')
    axs[0].set_xticks(x + bar_width * (len(models) - 1) / 2)
    axs[0].set_xticklabels([], rotation=45, fontname='Times New Roman')
    axs[0].set_ylim(40, 60)
    axs[0].legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=3, frameon=False)
    axs[0].text(0.5, 0.90, '0 Shot', fontsize=12, fontname='Times New Roman', ha='center', va='bottom', transform=axs[0].transAxes)

    # 第二子图
    for i, model in enumerate(models):
        model_data = data_2[data_2['Model'] == model]['Accuracy']
        axs[1].bar(x + i * bar_width, model_data, width=bar_width, label=model, color=colors[i], zorder=2)  # 设置柱状图的 zorder

    axs[1].set_xlabel('Subjects', fontsize=10, fontname='Times New Roman')
    axs[1].set_ylabel('Accuracy', fontsize=10, fontname='Times New Roman')
    axs[1].set_xticks(x + bar_width * (len(models) - 1) / 2)
    axs[1].set_xticklabels(subjects, rotation=0, fontname='Times New Roman')
    axs[1].set_ylim(40, 60)
    axs[1].text(0.5, 0.90, '5 Shot', fontsize=12, fontname='Times New Roman', ha='center', va='bottom', transform=axs[1].transAxes)
    # 设置 y 轴刻度标签为新罗马字体
    for tick in axs[0].get_yticklabels():
        tick.set_fontname('Times New Roman')
    for tick in axs[1].get_yticklabels():
        tick.set_fontname('Times New Roman')

    plt.tight_layout()
    plt.savefig('./fig/Fig.11.png', dpi=300)
    # plt.show()

def figs1():
    # 读取CSV文件
    data = pd.read_csv('D:\student\lzy\CDIP-ChatGLM3\data\LLM_Metric\sum_metrics.csv')

    # 获取所有作物和模型名称
    crops = data['Crop'].unique()
    models_to_exclude = ['freeze10_alpaca_10K', 'freeze10_alpaca_20K', 'freeze10_alpaca_5K']

    # 过滤掉不需要的模型
    data_filtered = data[~data['Model'].isin(models_to_exclude)]

    # 定义模型排序顺序
    model_order = ['chatglm6b', 'lora3', 'lora5', 'lora10', 'freeze3', 'freeze5', 'freeze10']

    # 设置字体属性
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 创建一个绘图，设置尺寸为（7, 10）
    fig, axs = plt.subplots(2, 1, figsize=(7, 8))

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#7f4c7a', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#f7b6d2', '#c5b0d5', '#ffbb78', '#ff9896', '#c49c94'
    ]

    # 创建颜色字典
    color_dict = {crop: colors[i % len(colors)] for i, crop in enumerate(crops) if crop != 'average'}
    color_dict['average'] = '#d62728'  # 将红色分配给average类别
    color_dict['average'] = '#000000'
    # 绘制每个作物的BLEU-4指数折线图
    for crop in crops:
        crop_data = data_filtered[data_filtered['Crop'] == crop]
        
        # 将模型按照定义的顺序排序
        crop_data['Model'] = pd.Categorical(crop_data['Model'], categories=model_order, ordered=True)
        crop_data = crop_data.sort_values('Model')
        
        if crop == 'average':
            # axs[0].plot(crop_data['Model'], crop_data['BLEU-4(‰)'], marker='^', label=crop, color=color_dict[crop], linewidth=2)
            # for i in range(len(crop_data)):
            #     model_name = crop_data['Model'].iloc[i]
            #     score_value = crop_data['BLEU-4(‰)'].iloc[i]
            #     axs[0].annotate(f'{score_value:.2f}', 
            #                     xy=(model_name, score_value),
            #                     xytext=(5, 5), 
            #                     textcoords='offset points',
            #                     fontsize=9,
            #                     color=color_dict[crop],
            #                     arrowprops=dict(arrowstyle='->', color=color_dict[crop]))
            pass
        else:    
            axs[0].plot(crop_data['Model'], crop_data['BLEU-4(‰)'], marker='o', label=crop, color=color_dict[crop])

    # 设置BLEU-4图表属性
    axs[0].set_ylabel('BLEU-4 Index(‰)')
    axs[0].set_xticklabels([])
    axs[0].legend(title='Crops', bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=5)
    axs[0].grid()

    # 绘制Average ROUGE F-score折线图
    for crop in crops:
        crop_data = data_filtered[data_filtered['Crop'] == crop]
        
        # 将模型按照定义的顺序排序
        crop_data['Model'] = pd.Categorical(crop_data['Model'], categories=model_order, ordered=True)
        crop_data = crop_data.sort_values('Model')

        if crop == 'average':
            # axs[1].plot(crop_data['Model'], crop_data['Average ROUGE F-score(‰)'], marker='^', label=crop, color=color_dict[crop], linewidth=2)
            # for i in range(len(crop_data)):
            #     model_name = crop_data['Model'].iloc[i]
            #     score_value = crop_data['Average ROUGE F-score(‰)'].iloc[i]
            #     axs[1].annotate(f'{score_value:.2f}', 
            #                     xy=(model_name, score_value),
            #                     xytext=(5, 5), 
            #                     textcoords='offset points',
            #                     fontsize=9,
            #                     color=color_dict[crop],
            #                     arrowprops=dict(arrowstyle='->', color=color_dict[crop]))
            pass
        else:
            axs[1].plot(crop_data['Model'], crop_data['Average ROUGE F-score(‰)'], marker='o', label=crop, color=color_dict[crop])

    # 设置ROUGE图表属性
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Average ROUGE F-score(‰)')
    axs[1].set_xticklabels(crop_data['Model'])
    axs[1].grid()
    for tick in axs[0].get_yticklabels():
        tick.set_fontname('Times New Roman')
    for tick in axs[1].get_yticklabels():
        tick.set_fontname('Times New Roman')
    # 保存并显示图表
    plt.tight_layout()
    plt.savefig('./fig/Fig.S1.png', dpi=300)
    # plt.show()

def figs2():
    data = pd.read_csv('D:\student\lzy\CDIP-ChatGLM3\data\LLM_Metric\sum_metrics.csv')

    # 获取所有作物和模型名称
    crops = data['Crop'].unique()
    models_to_exclude = ['chatglm6b', 'freeze3', 'freeze5','lora5','lora10','lora3']

    # 过滤掉不需要的模型
    data_filtered = data[~data['Model'].isin(models_to_exclude)]

    # 定义模型排序顺序
    model_order = ['freeze10', 'freeze10_alpaca_5K', 'freeze10_alpaca_10K', 'freeze10_alpaca_20K']

    # 设置字体属性
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 创建一个绘图，设置尺寸为（7, 8）
    fig, axs = plt.subplots(2, 1, figsize=(7, 8))

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#7f4c7a', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#f7b6d2', '#c5b0d5', '#ffbb78', '#ff9896', '#c49c94'
    ]

    # 创建颜色字典
    color_dict = {crop: colors[i % len(colors)] for i, crop in enumerate(crops) if crop != 'average'}
    # color_dict['average'] = '#d62728'  # 将红色分配给average类别
    # color_dict['average'] = '#000000'

    for crop in crops:
        crop_data = data_filtered[data_filtered['Crop'] == crop]
        
        # 将模型按照定义的顺序排序
        crop_data['Model'] = pd.Categorical(crop_data['Model'], categories=model_order, ordered=True)
        crop_data = crop_data.sort_values('Model')
        
        if crop == 'average':
            # axs[0].plot(crop_data['Model'], crop_data['BLEU-4(‰)'], marker='^', label=crop, color=color_dict[crop], linewidth=2)
            # for i in range(len(crop_data)):
            #     model_name = crop_data['Model'].iloc[i]
            #     score_value = crop_data['BLEU-4(‰)'].iloc[i]
            #     axs[0].annotate(f'{score_value:.2f}', 
            #                     xy=(model_name, score_value),
            #                     xytext=(5, 5), 
            #                     textcoords='offset points',
            #                     fontsize=9,
            #                     color=color_dict[crop],
            #                     arrowprops=dict(arrowstyle='->', color=color_dict[crop]))
            pass
        else:    
            axs[0].plot(crop_data['Model'], crop_data['BLEU-4(‰)'], marker='o', label=crop, color=color_dict[crop])

    # 设置BLEU-4图表属性
    axs[0].set_ylabel('BLEU-4 Index(‰)')
    axs[0].set_xticklabels([])
    axs[0].legend(title='Crops', bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=5)
    axs[0].grid()

    # 绘制Average ROUGE F-score折线图
    for crop in crops:
        crop_data = data_filtered[data_filtered['Crop'] == crop]
        
        # 将模型按照定义的顺序排序
        crop_data['Model'] = pd.Categorical(crop_data['Model'], categories=model_order, ordered=True)
        crop_data = crop_data.sort_values('Model')

        if crop == 'average':
            # axs[1].plot(crop_data['Model'], crop_data['Average ROUGE F-score(‰)'], marker='^', label=crop, color=color_dict[crop], linewidth=2)

            # # 遍历所有模型，标注每个点的值
            # for i in range(len(crop_data)):
            #     model_name = crop_data['Model'].iloc[i]
            #     score_value = crop_data['Average ROUGE F-score(‰)'].iloc[i]
            #     axs[1].annotate(f'{score_value:.2f}', 
            #                     xy=(model_name, score_value),
            #                     xytext=(5, 5), 
            #                     textcoords='offset points',
            #                     fontsize=9,
            #                     color=color_dict[crop],
            #                     arrowprops=dict(arrowstyle='->', color=color_dict[crop]))
            pass
        else:
            axs[1].plot(crop_data['Model'], crop_data['Average ROUGE F-score(‰)'], marker='o', label=crop, color=color_dict[crop])

    # 设置ROUGE图表属性
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Average ROUGE F-score(‰)')
    axs[1].set_xticklabels(crop_data['Model'])
    axs[1].grid()
    for tick in axs[0].get_yticklabels():
        tick.set_fontname('Times New Roman')
    for tick in axs[1].get_yticklabels():
        tick.set_fontname('Times New Roman')
    # 保存并显示图表
    plt.tight_layout()
    plt.savefig('./fig/Fig.S2.png', dpi=300)
    # plt.show()

if __name__ == "__main__":
    figs2()
