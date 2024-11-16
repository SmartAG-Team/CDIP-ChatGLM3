import os
import pandas as pd

# Directory containing the test results
directory = './data/LLM/LLM_Metric/CMMLU-Metric'

# Function to process and combine CSV files for a given shot type
def process_shot_type(shot_type):
    """
    Processes the specified shot type directory (e.g., '0shot', '5shot').
    Combines all CSV files within the directory, adds the model name to each row,
    and saves the combined data to a single CSV file.
    
    Args:
        shot_type (str): The shot type ('0shot' or '5shot').
    """
    for test in os.listdir(directory):
        if not test.endswith('.csv'):
            # Construct the path to the shot type directory (0shot or 5shot)
            test_path = os.path.join(directory, test, 'accuracy', 'summary')
            shot_path = os.path.join(test_path, shot_type)
            
            # List to store all dataframes
            combined_data = []
            
            # Iterate over all CSV files in the shot type directory
            for filename in os.listdir(shot_path):
                if filename.endswith('.csv'):
                    # Extract model name by removing the .csv extension
                    model_name = filename[:-4]
                    
                    # Construct the full path to the CSV file
                    file_path = os.path.join(shot_path, filename)
                    
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(file_path)
                    
                    # Add a new column for the model name
                    df['Model'] = model_name
                    
                    # Append the DataFrame to the combined data list
                    combined_data.append(df)
            
            # Concatenate all the DataFrames into a single DataFrame
            final_df = pd.concat(combined_data, ignore_index=True)
            
            # Reorder the columns to have 'Model' as the first column
            final_df = final_df[['Model'] + [col for col in final_df.columns if col != 'Model']]
            
            # Define the path to save the combined results
            save_path = os.path.join(test_path, f'results_{shot_type}.csv')
            
            # Save the combined DataFrame to a CSV file (overwrite if it exists)
            final_df.to_csv(save_path, index=False)
            print(f"Saved combined CSV for {shot_type}: {save_path}")

def merge_csv_files(csv_files, shot_value, output_file):
    merged_data = []
    i = 1
    # 遍历每个 CSV 文件并进行处理
    for file in csv_files:
        # 读取 CSV 文件
        data = pd.read_csv(file)
        
        # 添加 shot 列，设置为指定的 shot_value
        data['Shot'] = shot_value
        
        # 添加 Test 列，命名为 Test1, Test2, Test3...
        data['Test'] = f'Test{i}'
        i += 1
        
        # 将处理后的数据添加到列表中
        merged_data.append(data)
    
    # 将所有数据合并
    final_data = pd.concat(merged_data, ignore_index=True)
    
    # 保存合并后的数据到输出文件
    final_data.to_csv(output_file, index=False)
    print(f"CSV 文件已合并并保存到 {output_file}")

def merge_shots(shot0_files, shot5_files, final_output_file):
    # 合并 0shot 和 5shot 数据
    shot0_output = './data/LLM/LLM_Metric/CMMLU-Metric/0shot.csv'
    shot5_output = './data/LLM/LLM_Metric/CMMLU-Metric/5shot.csv'
    
    merge_csv_files(shot0_files, 0, shot0_output)
    merge_csv_files(shot5_files, 5, shot5_output)
    
    # 读取已合并的 0shot 和 5shot 数据
    shot0_data = pd.read_csv(shot0_output)
    shot5_data = pd.read_csv(shot5_output)
    
    # 合并 0shot 和 5shot 数据
    final_data = pd.concat([shot0_data, shot5_data], ignore_index=True)
    
    # 保存到最终输出文件
    final_data.to_csv(final_output_file, index=False)
    print(f"0shot 和 5shot 数据已合并并保存到 {final_output_file}")
    # 读取CSV文件
    file_path = final_output_file  # 替换为你的CSV文件路径
    df = pd.read_csv(file_path)

    # 按照指定的列顺序重新排序
    columns_order = ['Test', 'Subject', 'Shot', 'Model', 'Accuracy']
    df = df[columns_order]

    # 保存重新排序后的CSV文件
    df.to_csv(file_path, index=False)

    print("列已按指定顺序排列。")

# 指定三个 CSV 文件路径和最终输出文件名
shot0_csv_files = [
    './data/LLM/LLM_Metric/CMMLU-Metric/CMMLU-Test1/accuracy/summary/results_0shot.csv',
    './data/LLM/LLM_Metric/CMMLU-Metric/CMMLU-Test2/accuracy/summary/results_0shot.csv',
    './data/LLM/LLM_Metric/CMMLU-Metric/CMMLU-Test3/accuracy/summary/results_0shot.csv'
]
shot5_csv_files = [
    './data/LLM/LLM_Metric/CMMLU-Metric/CMMLU-Test1/accuracy/summary/results_5shot.csv',
    './data/LLM/LLM_Metric/CMMLU-Metric/CMMLU-Test2/accuracy/summary/results_5shot.csv',
    './data/LLM/LLM_Metric/CMMLU-Metric/CMMLU-Test3/accuracy/summary/results_5shot.csv'
]
final_output_file = './data/LLM/LLM_Metric/CMMLU-Metric/CMMLU_Metric.csv'

process_shot_type('0shot')
process_shot_type('5shot')
merge_shots(shot0_csv_files, shot5_csv_files, final_output_file)
