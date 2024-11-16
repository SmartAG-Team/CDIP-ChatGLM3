import os
import csv
from collections import defaultdict

def count_images_in_subfolders(src_folder, extensions=None):
    # 默认图片扩展名
    if extensions is None:
        extensions = {'.jpg', '.JPG','.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    # 使用 defaultdict 来存储每个子文件夹名及其对应的图片数量
    folder_image_count = defaultdict(int)

    # 遍历文件夹中的所有子文件夹
    for subdir in os.listdir(src_folder):
        subdir_path = os.path.join(src_folder, subdir)

        # 只处理文件夹
        if os.path.isdir(subdir_path):
            image_count = 0

            # 遍历每个子文件夹中的文件，计算图片数量
            for root, _, files in os.walk(subdir_path):
                for file in files:
                    # 检查文件扩展名是否是图片格式
                    if any(file.lower().endswith(ext) for ext in extensions):
                        image_count += 1

            # 累加子文件夹名对应的图片数量
            folder_image_count[subdir] += image_count

    return folder_image_count

def save_to_csv(data, output_csv):
    # 保存结果到 CSV 文件
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入标题
        writer.writerow(['Subfolder', 'Total Image Count'])
        # 写入每个子文件夹的合并图片数量
        for subdir, count in data.items():
            writer.writerow([subdir, count])

# 示例使用
src_folder = './data/CV/CV_dataset'  # 替换为你的文件夹路径
output_csv = './data/CV/CV_Metric/image_count.csv'  # 输出的 CSV 文件路径

# 统计 test 和 train-val 文件夹内子文件夹的图片数量
# 统计test文件夹
test_folder = os.path.join(src_folder, 'test')
test_image_counts = count_images_in_subfolders(test_folder)

# 统计train-val文件夹
train_val_folder = os.path.join(src_folder, 'train-val')
train_val_image_counts = count_images_in_subfolders(train_val_folder)

# 合并 test 和 train-val 文件夹中相同子文件夹名的图片数量
all_image_counts = defaultdict(int)

# 将 test 文件夹的统计结果加入合并字典
for subdir, count in test_image_counts.items():
    all_image_counts[subdir] += count

# 将 train-val 文件夹的统计结果加入合并字典
for subdir, count in train_val_image_counts.items():
    all_image_counts[subdir] += count

# 保存结果到 CSV 文件
save_to_csv(all_image_counts, output_csv)

print(f"图片数量已保存到 {output_csv}")
