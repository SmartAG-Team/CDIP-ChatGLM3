import os

# 设置目标文件夹和合并文件名
folder_path = r'D:\student\lzy\CDIP-ChatGLM3\data\LLM_books\文字数据集\葡萄文字分段\葡萄文字分段\第九章 葡萄病虫害防控常用农药简介\第一节 葡萄病害防控常用杀菌剂'  # 替换为你的文件夹路径
output_file = 'merged_output.txt'  # 合并文件名

# 打开合并文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    for filename in txt_files:  # 遍历所有txt文件
        file_path = os.path.join(folder_path, filename)
        # 写入文件名作为第一行
        outfile.write(f'{filename.replace(".txt", "")}\n')
        # 读取原文件内容并写入合并文件
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
            contents = infile.read()
            outfile.write(contents + '\n')

print('合并完成。')
