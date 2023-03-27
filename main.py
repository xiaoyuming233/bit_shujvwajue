import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib import pyplot as plt
from matplotlib import font_manager
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd


# 导入csv文件训练集
def loadData1(dataName):
    dataset = []
    with open(dataName, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for ii in reader:
            dataset.append(ii)
    return dataset


def storePreData(data, fileName):
    csvfile = open(fileName, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerows(data)
    csvfile.close()


# 设置盒图的图片参数
def draw_box(all_data):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    data = all_data
    df = pd.DataFrame(data)  # 读取数据
    df.plot.box(title="数据箱型图")  # 设置标题
    plt.grid(linestyle="--", alpha=0.3)  # 绘制箱型图
    plt.show()


def draw_histogram(a):
    # 计算组数
    d = 15  # 组数
    num_bins = (max(a) - min(a)) // d

    # 设置图形大小
    plt.figure(figsize=(20, 8), dpi=80)
    plt.hist(a, num_bins, density=True)
    # 使用density实现频率化

    # 设置x轴刻度
    plt.xticks(range(min(a), max(a) + d, d))

    # 设置网格
    plt.grid(alpha=0.4)
    plt.show()


length = 15
judge = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 2, 2, 1, 0]
count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
info = loadData1("movies_dataset.csv")
title = info[0]
storage = []
for i in range(length):
    storage.append([])
for i in info[1:]:
    for j in range(length):
        if len(i[j]) == 0:
            count[j] += 1
        else:
            if judge[j] == 1:
                temp = i[j].split(",")
                nowStr = "".join(temp)
                temp = nowStr.split("h")
                if len(temp) == 1:
                    temp1 = "".join(list(filter(str.isdigit, str(temp[0]))))
                    storage[j].append(float(temp1))
                else:
                    temp1 = "".join(list(filter(str.isdigit, str(temp[1]))))
                    if len(temp1) == 0:
                        temp1 = 0
                    storage[j].append(int(temp[0]) * 60 + int(temp1))
            else:
                storage[j].append(i[j])
boxInfo = []
for i in range(1, length):
    nowInfo = storage[i]
    nowJudge = judge[i]
    if nowJudge == 1:
        mini = min(nowInfo)
        a1 = np.percentile(nowInfo, 25)
        a2 = np.percentile(nowInfo, 50)
        a3 = np.percentile(nowInfo, 75)
        maxi = max(nowInfo)
        print(mini, a1, a2, a3, maxi, count[i])
        print("==========================================================")
        # draw_box(nowInfo)
    elif nowJudge == 0:
        nowInfo.sort()
        remember = []
        name = []
        index = 0
        remember.append(1)
        name.append(nowInfo[0])
        for j in range(1, len(nowInfo)):
            if nowInfo[j] == nowInfo[j - 1]:
                remember[index] += 1
            else:
                index += 1
                remember.append(1)
                name.append(nowInfo[j])
        for j in range(len(name)):
            print(name[j], remember[j])
        print(count[i])
        print("==========================================================")
