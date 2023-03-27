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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


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


length = 54
count = []
for i in range(length):
    count.append(0)
info = loadData1("clinical_data.csv")
title = info[0]
storage = []
for i in range(length):
    storage.append([])
for i in info[1:]:
    for j in range(length):
        if len(i[j]) == 0:
            count[j] += 1
        else:
            storage[j].append(float(i[j]))
# for i in range(1, length):
#     nowInfo = storage[i]
#     mini = min(nowInfo)
#     a1 = np.percentile(nowInfo, 25)
#     a2 = np.percentile(nowInfo, 50)
#     a3 = np.percentile(nowInfo, 75)
#     maxi = max(nowInfo)
#     print(title[i], mini, a1, a2, a3, maxi, count[i])
#     print("==========================================================")
#     draw_box(nowInfo)

nowInfo = storage[27]
mini = min(nowInfo)
a1 = np.percentile(nowInfo, 25)
a2 = np.percentile(nowInfo, 50)
a3 = np.percentile(nowInfo, 75)
maxi = max(nowInfo)
print(title[27], mini, a1, a2, a3, maxi, count[27])
print("==========================================================")
draw_box(nowInfo)
#
# for i in range(count[15]):
#     nowInfo.append(2)
# mini = min(nowInfo)
# a1 = np.percentile(nowInfo, 25)
# a2 = np.percentile(nowInfo, 50)
# a3 = np.percentile(nowInfo, 75)
# maxi = max(nowInfo)
# print(mini, a1, a2, a3, maxi)
# print("==========================================================")
# draw_box(nowInfo)
# data = pd.read_csv("clinical_data.csv")
# m, n = data.shape
# data = data.iloc[0:m + 1, 1:]
# # 计算相关系数矩阵
# data = data.corr()
# # 处理中文乱码
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 坐标轴负号的处理
# plt.rcParams['axes.unicode_minus'] = False
# mask = np.zeros_like(data)
# for i in range(1, len(mask)):
#     for j in range(0, i):
#         mask[j][i] = True
# # 设置下三角mask遮罩，上三角将i,j互换即可
# sns.heatmap(data=data, mask=mask, cmap='RdBu', vmax=1, vmin=-1, center=0, annot=True, square=True, linewidths=0,
#             cbar_kws={"shrink": .6}, xticklabels=True, yticklabels=True, fmt='.2f')
# # center 值越大颜色越浅
# # shrink:n n为缩短的比例(0-1)
# # fmt='.2f' 显示数字保留两位小数
# plt.title('热力图', fontsize='xx-large', fontweight='heavy')
# # 设置标题字体
# plt.show()


# m = len(storage[27])
# sum_x = np.sum(storage[27])
# sum_y = np.sum(storage[28])
# sum_xy = np.sum(storage[27] * storage[28])
# sum_xx = np.sum(storage[27] ** 2)
# a = (sum_y * sum_xx - sum_x * sum_xy) / (m * sum_xx - (sum_x) ** 2)
# b = (m * sum_xy - sum_x * sum_y) / (m * sum_xx - (sum_x) ** 2)
# print(a, b)


# def set_missing_ages(df):
#
#     # 把已有的数值型特征取出来丢进Random Forest Regressor中
#     age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
#
#     # 乘客分成已知年龄和未知年龄两部分
#     known_age = age_df[age_df.Age.notnull()].as_matrix()
#     unknown_age = age_df[age_df.Age.isnull()].as_matrix()
#
#     # y即目标年龄
#     y = known_age[:, 0]
#
#     # X即特征属性值
#     X = known_age[:, 1:]
#
#     # fit到RandomForestRegressor之中
#     rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
#     rfr.fit(X, y)
#
#     # 用得到的模型进行未知年龄结果预测
#     predictedAges = rfr.predict(unknown_age[:, 1:])
# #     print predictedAges
#     # 用得到的预测结果填补原缺失数据
#     df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges
#
#     return df, rfr

# data_train['CabinCat'] = data_train['Cabin'].copy()
# data_train.loc[ (data_train.CabinCat.notnull()), 'CabinCat' ] = "No"
# data_train.loc[ (data_train.CabinCat.isnull()), 'CabinCat' ] = "Yes"
#
# fig, ax = plt.subplots(figsize=(10,5))
# sns.countplot(x='CabinCat', hue='Survived',data=data_train)
# plt.show()

# Python中的使用：
# #使用price均值对NA进行填充
# df['price'].fillna(df['price'].mean())
# df['price'].fillna(df['price'].median())