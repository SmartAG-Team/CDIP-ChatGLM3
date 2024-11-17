import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_responses(data_source,output_file_path):
    tokenizer = AutoTokenizer.from_pretrained(
        "./model/LLM_models/Freeze/Freeze10(S2.5K)-DMT(S2.5K+G2.5K)/sft", 
        trust_remote_code=True
    )
    gpt_model = AutoModelForCausalLM.from_pretrained(
        "./model/LLM_models/Freeze/Freeze10(S2.5K)-DMT(S2.5K+G2.5K)/sft", 
        trust_remote_code=True, 
        device='cuda'
    )
    gpt_model = gpt_model.to('cuda').eval()
    responses = []
    for item in data_source:
        instruction = item["instruction"]
        response, _ = gpt_model.chat(tokenizer, instruction, history=[])
        entry = {"instruction": instruction, "input": "", "output": response}
        responses.append(entry)
        print(entry)
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(responses, json_file, ensure_ascii=False, indent=4)
    print("生成的回复已保存")

    return responses

if __name__ == "__main__":

    directory = './data/LLM/LLM_dataset/13-Crop-Instruction-Following-Dataset/summary'
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            pingzhong = file
            name = pingzhong.replace('.json','')

            response_file_path = f'./data/LLM/LLM_Model_Response/Freeze10(S2.5K)-DMT(S2.5K+G2.5K)/response_{file}'
            response_dir = os.path.dirname(response_file_path)
            if not os.path.exists(response_dir):
                os.makedirs(response_dir)
            print(file_path)
            print(response_file_path)
            print('-------------')
            with open(file_path, "r", encoding="utf-8") as json_file:
                data_source_1 = json.load(json_file)
            generate_responses(data_source_1,response_file_path)