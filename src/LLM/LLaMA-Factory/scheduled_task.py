import time
import subprocess

# # 5 小时的秒数
# hours = 5.5
# delay = hours * 60 * 60

# # 延迟5小时
# print(f"Waiting for {hours} hours before executing the command...")
# time.sleep(delay)



# 定义命令
command2 = "llamafactory-cli train llama3_freeze_sft_2.yaml"

# 执行命令（在新终端中）
try:
    subprocess.run(command2, shell=True, check=True)
    print("Command executed successfully in a new terminal window!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")

# 定义命令
command3 = "llamafactory-cli train llama3_freeze_sft_3.yaml"

# 执行命令（在新终端中）
try:
    subprocess.run(command3, shell=True, check=True)
    print("Command executed successfully in a new terminal window!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")

    # 定义命令
command4 = "llamafactory-cli train llama3_lora_sft.yaml"

# 执行命令（在新终端中）
try:
    subprocess.run(command4, shell=True, check=True)
    print("Command executed successfully in a new terminal window!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")

    # 定义命令
command5 = "llamafactory-cli train llama3_lora_sft_2.yaml"

# 执行命令（在新终端中）
try:
    subprocess.run(command5, shell=True, check=True)
    print("Command executed successfully in a new terminal window!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")

    # 定义命令
command6 = "llamafactory-cli train llama3_lora_sft_3.yaml"

# 执行命令（在新终端中）
try:
    subprocess.run(command6, shell=True, check=True)
    print("Command executed successfully in a new terminal window!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")