import os
import re
import numpy as np
import argparse

def extract_performance(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        # 使用正则表达式找到最后一次出现“The best performance:”后的数值
        matches = re.findall(r'The best performance:(\d+\.\d+)', content)
        if matches:
            return float(matches[-1])
        else:
            return None

def main(directory_path, multi_exp=False):
    performances = []
    if multi_exp:
        for exp in os.listdir(directory_path):
            file_path = os.path.join(directory_path, exp, "log.txt")
            if os.path.exists(file_path):
                performance = float(extract_performance(file_path))
                print(f"Accuracy of {file_path} is {performance}")
                if performance is not None:
                    performances.append(performance)
    else:
        file_path = os.path.join(directory_path, "log.txt")
        if os.path.exists(file_path):
            performance = float(extract_performance(file_path))
            print(f"Accuracy of {file_path} is {performance}")
            if performance is not None:
                performances.append(performance)

    # 计算平均值和方差
    if performances:
        average_performance = np.mean(performances)
        variance_performance = np.std(performances)
        print(f"Average  of {len(performances)} files is {average_performance:.2f} +- {variance_performance:.2f}")
    else:
        print("No valid performances found in the specified directory.")

parser = argparse.ArgumentParser()
parser.add_argument('exp', type=str)
parser.add_argument('--multi', action='store_true')
args = parser.parse_args()

# 指定包含txt文件的目录路径
directory_path = args.exp
main(directory_path, args.multi)
