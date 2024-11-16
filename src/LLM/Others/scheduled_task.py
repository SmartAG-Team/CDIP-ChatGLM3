import time
import subprocess

co = "python D:\student\lzy\CDIP-ChatGLM3\src\generate_and_score.py"

try:
    subprocess.run(co, shell=True, check=True)
    print("Command executed successfully")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")

command = "python D:\student\lzy\CDIP-ChatGLM3\src\CMMLU\src\chatglm.py"

try:
    subprocess.run(command, shell=True, check=True)
    print("Command executed successfully")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")

command1 = "python D:\student\lzy\CDIP-ChatGLM3\src\CMMLU\src\chatglm1.py"

try:
    subprocess.run(command1, shell=True, check=True)
    print("Command executed successfully")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")

command2 = "python D:\student\lzy\CDIP-ChatGLM3\src\CMMLU\src\chatglm2.py"

try:
    subprocess.run(command2, shell=True, check=True)
    print("Command executed successfully")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")

command3 = "python D:\student\lzy\CDIP-ChatGLM3\src\CMMLU\src\chatglm3.py"

try:
    subprocess.run(command3, shell=True, check=True)
    print("Command executed successfully")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")

command4 = "python D:\student\lzy\CDIP-ChatGLM3\src\CMMLU\src\chatglm4.py"

try:
    subprocess.run(command4, shell=True, check=True)
    print("Command executed successfully")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")

command5 = "python D:\student\lzy\CDIP-ChatGLM3\src\CMMLU\src\chatglm5.py"

try:
    subprocess.run(command5, shell=True, check=True)
    print("Command executed successfully")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e}")


