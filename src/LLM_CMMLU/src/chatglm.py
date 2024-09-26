import os
import torch
import numpy as np
import argparse
from mp_utils import choices, format_example, gen_prompt, softmax, run_eval
import pandas as pd  # 导入pandas库
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def eval_chat(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot):
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        prompt = gen_prompt(dev_df=dev_df,
                            subject=subject,
                            prompt_end=prompt_end,
                            num_few_shot=num_few_shot,
                            tokenizer=tokenizer,
                            max_length=max_length,
                            cot=cot)
        label = test_df.iloc[i, test_df.shape[1] - 1]

        pred, history = model.chat(tokenizer, prompt, history=[])
        if pred and pred[0] in choices:
            cors.append(pred[0] == label)
        all_preds.append(pred.replace("\n", ""))

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    result_df = pd.DataFrame({"Subject": [subject], "Accuracy": [round(acc, 3)]})
    if not os.path.isfile(args.csv_dir):
        result_df.to_csv(args.csv_dir, index=False)
    else:
        result_df.to_csv(args.csv_dir, mode='a', header=False, index=False)
    
    print("{} results, {} inappropriate formated answers.".format(len(cors), len(all_preds)-len(cors)))
    return acc, all_preds, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=r"D:\student\lzy\llM\LLaMA-Factory-main\saves\freeze\freeze_10_all\sft")
    parser.add_argument("--lora_weights", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./src/LLM_CMMLU/data")
    parser.add_argument("--save_dir", type=str, default="./data/LLM_CMMLU/results/freeze_10_all")
    parser.add_argument("--csv_dir", type=str, default="./data/LLM_CMMLU/results/freeze_10_all_0shot.csv")
    parser.add_argument("--sum_csv_dir", type=str, default="./data/LLM_CMMLU/results/freeze_10_all_sum_0shot.csv")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--load_in_8bit", action='store_true')
    parser.add_argument("--cot", action='store_true')
    args = parser.parse_args()

    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True,)
    model = AutoModel.from_pretrained(args.model_name_or_path,
                                      trust_remote_code=True,
                                      load_in_8bit=args.load_in_8bit,
                                    ).half().cuda()
    
    # Always use Chat-style evaluation
    run_eval(model, tokenizer, eval_chat, args)
