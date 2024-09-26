import matplotlib.pyplot as plt

# 数据
epochs = [2, 3, 6, 8, 10]
bleu_scores = [6.924, 7.054, 8.481, 8.688, 8.537]
avg_rouge_scores = [9.539, 23.06, 359.4, 551.2, 614.8]
dialogue_scores = [50, 28, 24, 24, 14]

# 创建子图
fig, axs = plt.subplots(3, 1, figsize=(7, 12))  # 调整子图高度

# 第一个子图：BLEU-4 score
axs[0].plot(epochs, bleu_scores, marker='o', linestyle='-', color='blue')
axs[0].set_title('BLEU-4 score', fontsize=14)
axs[0].set_ylabel('BLEU-4 score (‰)', fontsize=12)
for epoch, bleu in zip(epochs, bleu_scores):
    axs[0].text(epoch, bleu, f"{bleu:.3f}", ha='center', va='bottom', fontsize=10, rotation=15)

# 第二个子图：Avg ROUGE F-score
axs[1].plot(epochs, avg_rouge_scores, marker='o', linestyle='-', color='green')
axs[1].set_title('Avg ROUGE F-score', fontsize=14)
axs[1].set_ylabel('Avg ROUGE F-score (‰)', fontsize=12)
for epoch, rouge in zip(epochs, avg_rouge_scores):
    axs[1].text(epoch, rouge, f"{rouge:.1f}", ha='center', va='bottom', fontsize=10, rotation=15)

# 第三个子图：一半对话能力评分
axs[2].plot(epochs, dialogue_scores, marker='o', linestyle='-', color='red')
axs[2].set_title('General language proficiency score', fontsize=14)
axs[2].set_ylabel('General language proficiency score', fontsize=12)
for epoch, dialogue in zip(epochs, dialogue_scores):
    axs[2].text(epoch, dialogue, f"{dialogue}", ha='center', va='bottom', fontsize=10, rotation=15)

# 设置x轴标签和范围
for ax in axs:
    ax.set_xticks(epochs)
    ax.set_xticklabels(epochs)
    ax.set_xlim([1, 11])

# 设置x轴标签
axs[2].set_xlabel('Epoch', fontsize=12)

# 设置y轴标签格式和单位
for ax in axs:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)

# 设置y轴刻度字体大小
for ax in axs:
    ax.tick_params(axis='y', labelsize=10)

# 调整子图之间的间距
plt.tight_layout()

# 保存图片
plt.savefig('zhe.png', dpi=300)

# 显示图形
plt.show()
