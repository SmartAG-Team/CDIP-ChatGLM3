import os
import json
import csv
import hashlib
directory = './data/LLM/LLM_Model_Response'
# def hash_text(text):
#     return hashlib.md5(text.encode('utf-8')).hexdigest()

# def contains_repetitive_pattern(output_text, window_size=3, repetition_threshold=3):
#     words = output_text.split()
#     word_count = len(words)
    
#     if word_count < window_size * 2:
#         return False
    
#     seen_hashes = {}
    
#     for i in range(word_count - window_size + 1):
#         current_window = ' '.join(words[i:i + window_size])
#         current_hash = hash_text(current_window)
        
#         if current_hash in seen_hashes:
#             seen_hashes[current_hash] += 1
#             if seen_hashes[current_hash] >= repetition_threshold:
#                 return True
#         else:
#             seen_hashes[current_hash] = 1
    
#     return False

def process_folder(directory):
    model_error_rates = {}
    for root, dirs, files in os.walk(directory):
        model_name = os.path.basename(root)  # 初始化 model_name
        error_count = 0
        total_count = 0
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as json_file:
                    data_source = json.load(json_file)
                    total_count += len(data_source)
                    for i in data_source:
                        output_text = i.get('output', '')
                        if len(output_text) > 5000:
                            error_count += 1
        if total_count > 0:  # 确保不会除以零
            error_rate = (error_count / total_count) * 100
            model_error_rates[model_name] = error_rate
    return model_error_rates

error_rates = process_folder(directory)

csv_filename = './data/LLM/LLM_Metric/Error-rate-Metric/Error-rate-Metric-str.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Model', 'Error Rate (%)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for model_name, error_rate in error_rates.items():
        writer.writerow({
            'Model': model_name,
            'Error Rate (%)': f"{error_rate:.2f}"
        })

print(f"Error rate information has been saved to {csv_filename}")
