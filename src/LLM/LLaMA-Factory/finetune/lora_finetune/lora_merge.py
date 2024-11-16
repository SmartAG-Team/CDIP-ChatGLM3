import os
import subprocess
import yaml

current_directory = os.getcwd()
config_path = os.path.join(current_directory, 'model', 'LLM_Config', 'LoRA','LoRA_merge.yaml')
model_path = os.path.join(current_directory, 'model', 'LLM_Models', 'ChatGLM3_6B')
adapter_dir = os.path.join(current_directory, 'model', 'LLM_Models', 'LoRA_before_merge', 'LoRA3')
output_dir = os.path.join(current_directory, 'model', 'LLM_Models','LoRA3')

llama_factory_directory = os.path.join(current_directory, 'src', 'LLM', 'LLaMA-Factory')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
config['model_name_or_path'] = model_path
config['output_dir'] = output_dir
with open(config_path, 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

command = f"llamafactory-cli export {config_path}"

try:
    subprocess.run(command, cwd=llama_factory_directory, shell=True, check=True)
    print("Command executed successfully in a new terminal window!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")