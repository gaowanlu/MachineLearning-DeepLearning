from email import header
import pandas as pd
import numpy as np
from pyparsing import col

# 数据准备 加载数据集
HR_DATA_PATH = "./HR-Employee-Attrition.csv"


def load_data(path):
    return pd.read_csv(path)


hr_data = load_data(HR_DATA_PATH)


def data_format():
    # 读取表头
    headers = list(pd.read_csv(HR_DATA_PATH))
    # 读取数据
    row_list = pd.read_csv(HR_DATA_PATH, usecols=headers).values.tolist()
    col_list = pd.read_csv(HR_DATA_PATH, usecols=headers).T.values.tolist()
    return headers, row_list, col_list


headers, row_list, col_list = data_format()
# print(headers)
# print(row_list)
print("数据集共有样本数目为 {}".format(len(row_list)))


str_col_index = []
for i in range(len(row_list[0])):
    if isinstance(row_list[0][i], str):
        str_col_index.append(i)

print(str_col_index)

col_numlized = []
for i in range(len(col_list)):
    v = pd.Categorical(col_list[i]).codes
    col_numlized.append(v)
print(col_numlized)

# 得到字符串属性与数字编号
# 遍历原本为字符串的列
# for i in range(len(str_col_index)):
