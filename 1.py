import os

def merge_txt_files(folder_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write('\n')  # 添加换行符以分隔文件内容

if __name__ == '__main__':
    folder_path = r'D:\student\lzy\CDIP-ChatGLM3\data\LLM_books_2\文字数据集\葡萄文字分段\葡萄文字分段\第八章 葡萄主要病虫害防控技术\第一节 主要病害防控技术'  # 替换为你的文件夹路径
    output_file = 'D:\student\lzy\CDIP-ChatGLM3\data\LLM_books_2\文字数据集\葡萄文字分段\葡萄文字分段\import\主要病害防控技术.txt'  # 输出文件名
    merge_txt_files(folder_path, output_file)
