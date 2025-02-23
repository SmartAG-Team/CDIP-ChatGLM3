import os
import random
import shutil

def extract_random_images(source_folder, target_folder, num_images=10):
    """
    从 source_folder 的每个子文件夹中随机抽取 num_images 张图片，并复制到 target_folder 中。
    :param source_folder: 源文件夹，包含多个子文件夹，每个子文件夹内含图片
    :param target_folder: 目标测试数据集文件夹
    :param num_images: 每个子文件夹中要随机抽取的图片数量（默认 10 张）
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for subdir in os.listdir(source_folder):
        subdir_path = os.path.join(source_folder, subdir)
        
        if os.path.isdir(subdir_path):  # 确保是文件夹
            images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if len(images) < num_images:
                print(f"警告：'{subdir}' 文件夹中图片数量不足 {num_images} 张，仅有 {len(images)} 张，将全部复制。")
                selected_images = images  # 如果图片少于 num_images，则全部复制
            else:
                selected_images = random.sample(images, num_images)  # 随机抽取 num_images 张图片
            
            # 在目标文件夹中创建相应的子文件夹
            target_subdir_path = os.path.join(target_folder, subdir)
            os.makedirs(target_subdir_path, exist_ok=True)
            
            # 复制图片到目标文件夹
            for img in selected_images:
                src_img_path = os.path.join(subdir_path, img)
                dst_img_path = os.path.join(target_subdir_path, img)
                shutil.copy2(src_img_path, dst_img_path)
    
    print(f"已完成图片抽取，测试数据存放于: {target_folder}")


source_folder = "./data/CV/CV_dataset/test"  # 替换为你的源文件夹路径
target_folder = "./data/CV/Image Understanding Test Dataset"  # 替换为你的目标测试数据集文件夹
extract_random_images(source_folder, target_folder)
