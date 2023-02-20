# -*-coding:utf-8-*-

"""
（1）编码问题
（2）缺失值的检测与处理
（3）去除异常值
（4）去重重复值与冗余信息
（5）操作日志记录
"""

# 数据编码
import numpy as np
import requests

data = requests.get('http://www.baidu.com')
# print(data.text)
# 中文未被正常显示，显然存在编码问题。使用ftfy库可以解决乱码问题
from ftfy import fix_text
# print(fix_text(data.text))


# 缺失值的检测与处理
from pandas import DataFrame

# df = DataFrame({'c1': [0, 1, 2, None], 'c2': [1, None, 2, 3]})
# print(df)
# print(df.isnull())
# print(df.fillna('missing'))
# print(df.fillna(df.mean()))
# # bfi11 意为使用缺失值后(下)面的值进行填充
# print(df.fillna(method='bfill', limit=1))
# # bfi11 意为使用缺失值前(上)面的值进行填充
# print(df.fillna(method='ffill', limit=1))

df = DataFrame({'01': np.arange(0, 0.5, 0.1), '02': np.arange(1, 1.5, 0.1)})
# print(df)
# 设定指定位置的值  df.iloc（）可以在指定位置插入或更改数据。
df.iloc[1:3, 0:2] = None
df['03'] = np.nan  # 添加新的列03,并且内容都为空NaN
# print(df)
# df.dropna（）可以快速批量地删除数据
# print(df.dropna())  # 删除所有的数据
# print(df.dropna(how='all'))  # 删除了含有NaN的行
# print(df.dropna(how='all', axis=1))  # 删除了含有NaN的列
# df.iloc[0, 0] = 100
# print(df)


# 去除异常值
from pandas import DataFrame, Series

# df.query（）来筛选并去掉异常值
df = DataFrame({'Name': ['A', 'B', 'C'], 'Age': [- 1, 1, 125]})
# print(df)
# print(df.query('Age > = 10 and Age < = 110'))

df_dict = DataFrame({'Age': [16, 17, 20, 21, 22], 'Age_label': ['teen', 'adult', 'adult', 'adult', 'teen']})
# print(df_dict)
df_data = df_dict.query("(Age>= 18 and Age_label == 'adult') or (Age< 18 and Age_label =='teen')")
# print(df_data)

import matplotlib.pyplot as plt

# 通过箱型图进而确定异常点
# series_data = Series([3, 1, 5, 7, 10, 50])
# plt.boxplot(series_data)
# plt.show()


# 去重重复值与冗余信息
"""
df.drop_duplicates（）常用的参数有subset、keep、inplace，
用法如下：
·　subset，默认值为None，默认对整个DataFrame进行去重，可以通过指定的subset对特定的列去重。
·　keep，默认值为first，即在有重复值的时候只保留第一次出现的数据。除此之外，可以选择last来保留最后出现的数据；如果选择False，则会去除所有重复的数据。
·　inplace，默认值为False，当其值为False时，不改变原来的DataFrame，返回新筛选后的数据。
"""
from pandas import DataFrame

df = DataFrame({'A': [1, 1, 2, 2], 'B': [3, 3, 4, 4]})
# print(df)
# print(df.drop_duplicates())


# 其他注意事项
"""
注意数据的备份和处理流程的记录。
在处理较为庞大的数据集时，无论对数据分析的严谨性来说，还是对团队合作来说，对处理的流程进行适当记录都是十分必要的。
根据具体的需要自定义日志记录的格式
"""
import os
import time
import datetime
import pandas as pd


# 获取日期和时间
def get_date_and_time():
    # 获取时间戳
    timestamp = time.time()
    # 将时间截转化为指定格式的时间格式
    value = datetime.datetime.fromtimestamp(timestamp)
    date_and_time = value.strftime('%Y-%m-%d %H:%M:%S')
    return date_and_time


# 日志文件操作
def write_to_log(logname='Report.txt', oprations=None):  # 检查是否创建了日志文件

    if logname not in os.listdir():
        with open(logname, 'w')as f:
            # 创建文件
            f.writelines(["My Report -- Created by Yu ", get_date_and_time()])
            f.write("\n")
            # 写入数据
            f.writelines([get_date_and_time(), ': '])
            f.write(oprations)
            f.write("\n")
    else:
        # 已有日志文件，则以追加的模式写人记录
        with open(logname, 'a') as f:
            # 以追加模式写人数据
            f.writelines([get_date_and_time(), ': '])
            f.write(oprations)
            f.write("\n")
# if __name__ == '__main__':
#     write_to_log(oprations='Read data from result.csv')
#     df = pd.read_csv('result.csv')
#
#     write_to_log(oprations='drop the duplicate data')
#     df = df.drop_duplicates()