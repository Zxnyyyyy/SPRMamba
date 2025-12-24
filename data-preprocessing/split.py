import os
import csv
from collections import defaultdict
import random

# 文件夹路径
folder_path = "./datasets/ESD385/groundTruth/"

# 统计病例
cases = defaultdict(list)

# 遍历文件夹
for file_name in os.listdir(folder_path):
    case_id = file_name[:29]  # 提取前29个字符作为病例ID
    cases[case_id].append(file_name)

# 打乱病例顺序
random.seed(42)  # 设置随机种子，保证每次运行的结果相同
cases_list = list(cases.values())
random.shuffle(cases_list)

# 分配病例到不同的数据集
total_cases = len(cases_list)
print(total_cases)
train_end = int(0.7 * total_cases)
val_end = int(0.8 * total_cases)

# 写入CSV文件
def write_csv(file_name, data):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for case_files in data:
            for file_name in case_files:
                writer.writerow([file_name])

write_csv('./datasets/splits/train.csv', cases_list[:train_end])
write_csv('./datasets/splits/val.csv', cases_list[train_end:val_end])
write_csv('./datasets/splits/test.csv', cases_list[val_end:])