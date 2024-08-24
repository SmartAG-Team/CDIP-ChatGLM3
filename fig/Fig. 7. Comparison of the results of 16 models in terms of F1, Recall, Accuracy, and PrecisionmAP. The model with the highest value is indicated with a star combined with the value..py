import matplotlib.pyplot as plt
import numpy as np

# Model names
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
plt.savefig('performance_metrics_updated.png', dpi=300)
plt.show()
