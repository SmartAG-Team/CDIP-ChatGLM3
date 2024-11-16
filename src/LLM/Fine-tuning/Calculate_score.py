import os
import json
import csv
import jieba
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def calculate_score(data_candidate, data_reference, output_csv_path):
    """
    Calculate BLEU-4 and ROUGE scores for candidate and reference text data, then save the scores to a CSV file.

    Parameters:
    data_candidate (list): List of dictionaries with candidate output texts.
    data_reference (list): List of dictionaries with reference output texts.
    output_csv_path (str): Path to save the CSV file containing evaluation scores.

    Returns:
    tuple: Average ROUGE scores, BLEU-4 score, and average ROUGE F-score.
    """
    # Initialize ROUGE and total score dictionary
    rouge = Rouge()
    total_scores = {
        "rouge-1": {"f": 0, "p": 0, "r": 0},
        "rouge-2": {"f": 0, "p": 0, "r": 0},
        "rouge-l": {"f": 0, "p": 0, "r": 0},
        "bleu-4": 0
    }

    # Check if candidate and reference data lengths match
    if len(data_candidate) == len(data_reference):
        for item1, item2 in zip(data_candidate, data_reference):
            reference_text = item2.get("output", "")
            generated_text = item1.get("output", "")

            # Calculate ROUGE scores
            scores = rouge.get_scores(generated_text, reference_text)

            # Accumulate ROUGE scores
            for metric in scores[0]:
                for component in scores[0][metric]:
                    if metric in total_scores and component in total_scores[metric]:
                        total_scores[metric][component] += scores[0][metric][component]

            # Calculate BLEU-4 score with jieba tokenization
            reference = jieba.lcut(reference_text)
            candidate = jieba.lcut(generated_text)
            bleu_score = sentence_bleu(
                [reference], candidate, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=SmoothingFunction().method1
            )
            total_scores["bleu-4"] += bleu_score

        # Calculate average ROUGE and BLEU-4 scores
        num_samples = len(data_candidate)
        avg_scores = {
            metric: {component: total_scores[metric][component] / num_samples for component in total_scores[metric]}
            for metric in total_scores if metric != "bleu-4"
        }
        avg_bleu4 = total_scores["bleu-4"] / num_samples
        avg_f_score = (avg_scores["rouge-1"]["f"] + avg_scores["rouge-2"]["f"] + avg_scores["rouge-l"]["f"]) / 3
        print(f"Average BLEU-4 Score: {avg_bleu4}")
        print(f"Average ROUGE F-score: {avg_f_score}")

        # Write results to CSV
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Precision", "Recall", "F-score"])
            writer.writerow(["ROUGE-1", avg_scores["rouge-1"]["p"], avg_scores["rouge-1"]["r"], avg_scores["rouge-1"]["f"]])
            writer.writerow(["ROUGE-2", avg_scores["rouge-2"]["p"], avg_scores["rouge-2"]["r"], avg_scores["rouge-2"]["f"]])
            writer.writerow(["ROUGE-L", avg_scores["rouge-l"]["p"], avg_scores["rouge-l"]["r"], avg_scores["rouge-l"]["f"]])
            writer.writerow(["BLEU-4", "-", "-", avg_bleu4])
            writer.writerow(["Average ROUGE F-score", "-", "-", avg_f_score])

            print(f"Scores have been written to {output_csv_path}")

        return avg_scores, avg_bleu4, avg_f_score
    else:
        print("Mismatched data lengths between candidate and reference lists.")
        return None

if __name__ == "__main__":
    # Define paths for input data and output results
    directory = './data/LLM/LLM_dataset/13-Crop-Instruction-Following-Dataset/summary'
    response_directory = './data/LLM/LLM_Model_Response'
    folder_names = [name for name in os.listdir(response_directory) if os.path.isdir(os.path.join(response_directory, name))]

    # Loop through each model and calculate scores
    for model in folder_names:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                name = file.replace('.json', '')

                # Load candidate and reference data
                with open(file_path, "r", encoding="utf-8") as json_file:
                    data_source = json.load(json_file)
                with open(f"{response_directory}/{model}/response_{file}", "r", encoding="utf-8") as json_file:
                    finetune_source = json.load(json_file)

                # Define output CSV path and calculate scores
                csv_output_file_path = f'./data/LLM/LLM_Metric/Fine-tuning-Metric/{name}/{model}.csv'
                os.makedirs(os.path.dirname(csv_output_file_path), exist_ok=True)
                calculate_score(data_source, finetune_source, csv_output_file_path)

    # Collect results and summarize across all crops and models
    results = []
    metric_directory = './data/LLM/LLM_Metric/Fine-tuning-Metric'
    for crop in os.listdir(metric_directory):
        crop_path = os.path.join(metric_directory, crop)
        if os.path.isdir(crop_path):
            for model_file in os.listdir(crop_path):
                if model_file.endswith('.csv'):
                    file_path = os.path.join(crop_path, model_file)
                    df = pd.read_csv(file_path)

                    # Extract BLEU-4 and Average ROUGE F-score values
                    bleu_value = df.loc[df['Metric'] == 'BLEU-4', 'F-score'].values[0]
                    avg_rouge_value = df.loc[df['Metric'] == 'Average ROUGE F-score', 'F-score'].values[0]

                    # Format model name and scale scores
                    model_name = model_file.replace(".csv", "").replace("rouge_bleu_", "")
                    bleu_value = round(float(bleu_value) * 1000, 2)
                    avg_rouge_value = round(float(avg_rouge_value) * 1000, 2)

                    # Append results for current crop and model
                    results.append([crop, model_name, bleu_value, avg_rouge_value])

    # Create DataFrame, calculate averages, and save to CSV
    results_df = pd.DataFrame(results, columns=['Crop', 'Model', 'BLEU-4(‰)', 'Average ROUGE F-score(‰)'])
    results_df = results_df.dropna()
    average_results = results_df.groupby('Model', as_index=False).agg({
        'BLEU-4(‰)': 'mean',
        'Average ROUGE F-score(‰)': 'mean'
    })
    average_results['Crop'] = 'average'
    final_results = pd.concat([results_df, average_results], ignore_index=True)
    final_results.to_csv('./data/LLM/LLM_Metric/Fine-tuning-Metric/Fine-tuning-Metric.csv', index=False)
