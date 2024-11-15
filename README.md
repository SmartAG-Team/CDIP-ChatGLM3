# CDIP-ChatGLM3

**Crop disease identification and prescription using CNNs, ChatGLM3-6b, and Fine-tuned models**

https://github.com/user-attachments/assets/a098d6a3-00a9-4350-abe1-4187ef419887

https://github.com/user-attachments/assets/17621a5a-9d07-49d8-9760-ac859c511b9e

https://github.com/user-attachments/assets/e2cac48e-b38c-42f7-a41e-f4dcf88bf9b4

## Prerequisites

### Clone the Repository

Before starting with dataset preparation, clone the repository by running:

```
git clone git@github.com:SmartAG-Team/CDIP-ChatGLM3.git
```

Navigate to the project directory:

```
cd CDIP-ChatGLM3
```

### Environment Setup

The environment is developed with **Python 3.10**. To set up the environment, please install the required dependencies by running:

```
pip install -r requirements.txt
```

### Configuration

If you plan to create the instruction-following dataset yourself, you need to create a `.env` file in the working directory with the following configuration:

1. For Llama Model
   - Add `Llama_API_KEY` and `Llama_API_base_url`.
2. For Qwen Model
   - Add `Qwen_API_KEY` and `Qwen_API_base_url`.
3. For GPT Model
   - Add `OpenAI_API_KEY` and `OpenAI_API_base_url`.

## Model

### CV Models

Our trained CV models are stored in:
`(Provide the path here)`.

### LLM Model

Our trained LLM models are stored in:
[LLM Models (Baidu Pan Link)](https://pan.baidu.com/s/1CdN1sby8TCPNAS5W1aq5NQ?pwd=mbf4)
**Password**: mbf4

### Model Preparation:

Please place your models under the `./model` directory.

## Dataset Preparation

### 13-Crop disease Dataset

You need to place the disease dataset into the "CV" folder under the "data" directory.

### 13-Crop instruction-following Dataset

Path: `./data/LLM/LLM_dataset/13-Crop-Instruction-Following-Dataset` contains the 13-Crop instruction-following dataset.

Alternatively, you can create the dataset as follows:

- **Collection of Original Disease Control Books for 13 Crops**: (Provide source or link)

- **Manual Division, Cleaning, and Filtering**: Save the processed data to `./data/LLM/LLM_books`.

- **Constructing the 13-Crop Instruction-Following Dataset Using Llama3.1-405B-Instruct API**:

  To generate the corresponding instruction-following datasets for each crop's partitioned text information folder, use the script `./src/Fine-tuning/Create-dataset/create_dataset.py`. Specifically, the following parameters should be modified in the file:

  - `--folder_path`: The folder containing the partitioned chapter information.
  - `--prompt{1,2,3}_dir`: The folders containing the corresponding prompts for each crop. (Note: In `prompt1`, the crop name should be updated for each crop.)
  - `--instruction_number`: Specifies the number of instructions to generate for each folder.

  The instruction-following dataset generation process consists of three steps:

  - **Step 1 - `instruction_generation()`**: Generates a specified number of instructions based on the provided context.
  - **Step 2 - `original_output()`**: Generates raw answers based on the instructions.
  - **Step 3 - `output()`**: Generates refined and improved answers using the instructions, raw answers, and context.

-  After generating the instruction-following dataset, you can merge the data using the `./src/Fine-tuning/Create-dataset/merge_json.py` script.     Please ensure to modify the paths in the script to match the location of your data. This script will merge all the generated instruction data into a single, unified file for further processing or training.

### General Dataset: Alpaca

Path: `./data/LLM/LLM_dataset/General-Dataset-Alpaca` contains the Alpaca dataset and its data volume variant.

Alternatively, you can download the dataset from the following link: [llamafactory/alpaca_gpt4_zh at main](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh/tree/main)

### CMMLU Benchmark Dataset

Path: `./data/LLM/LLM_dataset/CMMLU-Benchmark-Dataset` contains the CMMLU benchmark dataset.

Alternatively, you can download the dataset from the following link: [haonan-li/CMMLU: Measuring massive multitask language understanding in Chinese](https://github.com/haonan-li/CMMLU)

### Specialized Abilities Test Dataset

Path: `./data/LLM/LLM_dataset/Specialized-Abilities-Test-Dataset/Specialized-Abilities-Test-Dataset.json` contains the Specialized Abilities Test dataset.

Alternatively, you can create the dataset as follows:

- **Extract 200 Disease Prevention and Control Issues** from specialized datasets using the following command:

  ```
  python ./src/Specialized-Abilities-Test/Dateset-create/Dataset_extraction.py
  ```

  This will extract and save 200 disease control test cases to `./data/LLM/LLM_dataset/Specialized-Abilities-Test-Dataset/Crop-Disease-200.json`.

- **Generate Change Questions for the Specialized Abilities Test Dataset** using the Llama3.1-405B-Instruct API:

  ```
  python ./src/Specialized-Abilities-Test/Dateset-create/Generate-Specialised-Abilities-Test-Dataset.py
  ```
## Train
### CV Training

Before starting the training, you need to place the disease dataset in the data/cv directory. Additionally, if you do not want to use the default model in the training code, you need to re-import and change the pretrained model. For example, with EfficientNet, you need to modify the following parts:
  ```
  from EfficientNet import efficientnet_b0 as create_model
  ```
 #### change
  ```
  from EfficientNet import efficientnet_b1 as create_model
  ```

And modify the default value of --weights in the main function to model\CV\pretrain_model\efficientnetb1.pth.Then you can start training.
## Results

### CV-Model Test

Our CV experiment results include four main components.

1. **Model Performance**: By comparing the Accuracy, Recall, F1-scores, and Precision of 10 models, we identify the best-performing model.
2. **Disease Identification Accuracy**: The prediction results are demonstrated by calculating the accuracy for each type of disease prediction.

####**Model Performance**
We selected 10 models for this experiment, which include:ResNet-34，ResNet-50，MobileNetV3-Small，EfficientNet-B0，EfficientNet-B1，EfficientNet-B2，EfficientNetV2-S，Swin-Transformer-Tiny，FasterNet-T0，FasterNet-T1.
To test the performance of the 10 models, you can run the following Python code：

  ```
  python src\CV\train-val\model\test.py
  ```

You should replace the 'model' in the command with the actual model being tested

####**Disease Identification Accuracy**
Our dataset includes the following crops: apple, cherry, citrus, corn, grape, peach, pepper, potato, rice, soybean, strawberry, tomato, and wheat. The full_confusion_matrix.csv contains the classification information for the diseases of all these crops. You can generate the table by running the code：

  ```
  python src\CV\train-val\efficientnet\test-csv.py
  ```

### Fine-Tuning Test

The results of our fine-tuning experiments consist of three main components:

1. **LLM-Generated Responses**: The answers generated by the fine-tuned LLM in response to the 13-crop instruction-following dataset questions.
2. **BLEU and ROUGE Scores**: These scores are calculated by comparing the generated answers with standard answers.
3. **Error Rate**: The error rate is defined as the percentage of responses that generated more than 5,000 tokens during repeated answer generation, which are considered errors.

#### **LLM-Generated Responses**

- The responses generated by the fine-tuned LLM are stored in `./data/LLM/LLM_Model_Response`. Each model provides answers to the questions in the 13-Crop dataset.

- To regenerate responses using the fine-tuned model, run:

  ```
  python ./src/Fine-tuning/Generate_LLM_Response.py
  ```

  This command uses the fine-tuned LLM to generate answers based on the dataset's questions.

- Since different fine-tuned models may respond to prompts in different ways, some responses may not adhere to the standard output format specified in the prompt. To ensure standardized formatting, run:

  ```
  python ./src/Fine-tuning/Trimming_LLM_Reply.py
  ```

  Ensure to manually adjust any paths as needed.

#### **BLEU and ROUGE Scores**

- The BLEU and ROUGE evaluation scores are stored in `./data/LLM/LLM_Metric/Fine-tuning-Metric`. These scores are calculated by comparing the LLM-generated responses with standard answers.

- To re-evaluate the fine-tuning experiment, ensure the fine-tuned responses are saved in `./data/LLM/LLM_Model_Response`, then run:

  ```
  python ./src/Fine-tuning/Calculate_score.py
  ```

#### **Error Rate**

- The error rate metric is stored in `./data/LLM/LLM_Metric/Error-rate-Metric`.

- To re-evaluate the error rate, ensure the fine-tuned responses are saved in `./data/LLM/LLM_Model_Response`, then run:

  ```
  python ./src/Fine-tuning/Error_Rate/Error_Rate_Token.py
  ```

### CMMLU Benchmark Test

Our CMMLU experiment results are stored in `./data/LLM/LLM_Metric/CMMLU-Metric`, with each experiment folder labeled as "Test{number}" to indicate the iteration. Each experiment folder contains Zero-shot and Five-shot accuracy metrics (`accuracy`) and detailed response results (`results`).

#### Validation

To generate evaluation metrics for CMMLU based on the experiment results, run:

```
python ./src/CMMLU/CMMLU_Get_Metrics.py
```

This script aggregates the CMMLU benchmark results and saves them to `./data/LLM/LLM_Metric/CMMLU-Metric/CMMLU_Metric.csv`.

#### Evaluation

To re-evaluate CMMLU using the model, run:

```
python ./src/CMMLU/chatglm.py
```

Make sure to manually adjust the model path, shot count, and perform multiple experiments as needed.

### Specialized Abilities Test

The results of our Specialized Abilities Test are stored in `./data/LLM/LLM_Metric/Specialized-Abilities-Test-Metric/LLM_Response`, where each LLM model has provided answers to all the questions in the test dataset.

#### Validation

To generate evaluation metrics for the Specialized Abilities Test based on the experiment results, run:

```
python ./src/Specialized-Abilities-Test/Specialized-Abilities-Test-Get-Metric.py
```

This script calculates the average BLEU and ROUGE scores for each model's responses and saves them to `./data/LLM/LLM_Metric/Specialized-Abilities-Test-Metric/Specialized-Abilities-Test-Metric.csv`.

#### Evaluation

To re-evaluate the specialized abilities in disease prevention and control for different LLM models, run the following commands:

- **ChatGLM3-6B**:

  ```
  python ./src/Specialized-Abilities-Test/Test/ChatGLM3-6B.py
  ```

- **Qwen-Max**:

  ```
  python ./src/Specialized-Abilities-Test/Test/qwen-max.py
  ```

- **Llama-3.1-405B-Instruct**:

  ```
  python ./src/Specialized-Abilities-Test/Test/llama3.1-405b-instruct.py
  ```

- **GPT-4o**:

  ```
  python ./src/Specialized-Abilities-Test/Test/gpt4o.py
  ```

- **CDIP-ChatGLM3**:

  ```
  python ./src/Specialized-Abilities-Test/Test/CDIP-ChatGLM3.py
  ```

After generating results for each model, run:

```
python ./src/Specialized-Abilities-Test/Specialized-Abilities-Test-Metric.py
```

This will update the metrics in `./data/LLM/LLM_Metric/Specialized-Abilities-Test-Metric/Specialized-Abilities-Test-Metric.csv`.

### Figs Generation

After collecting the following five data files

- `./data/CV/results.csv`
- `./data/LLM/LLM_Metric/CMMLU-Metric/CMMLU_Metric.csv`
- `./data/LLM/LLM_Metric/Error-rate-Metric/Error-rate-Metric-token.csv`
- `./data/LLM/LLM_Metric/Fine-tuning-Metric/Fine-tuning-Metric.csv`
- `./data/LLM/LLM_Metric/Specialized-Abilities-Test-Metric/Specialized-Abilities-Test-Metric.csv`

,you can generate all the data images using the provided code:

```
python ./src/All_fig.py
```

Alternatively, if you wish to generate specific images, you can modify the `fig{num}()` function in the script to specify which image you would like to generate.

Please ensure that all the required data files are available before running the image generation code. All generated images, including those created using Vision, will be stored in the `./fig` directory for easy access and further use.
