# -*-coding:utf-8-*-

# 导人需要的包
import os
import pandas as pd
import xlsxwriter

# 创建和打开Excel文件所在的路径
if 'myXlsxFolder' not in os.listdir():
    os.mkdir('myXlsxFolder')
os.chdir('myXlsxFolder')
# 将result.csv的数据按需求导人指定的Excel表格#输入爬取的数据
books_data = pd.read_csv('../result.csv', usecols=['name', 'author', 'publisher', 'price', 'rate', 'comment', 'URL'],
                         na_values='NULL')
titles = books_data['name']
authors = books_data['author']
publishers = books_data['publisher']
prices = books_data['price']
ratings = books_data['rate']
comments = books_data['comment']
URLs = books_data['URL']
# 创建电子表格文件book_info.xlsx并为其添加一个名为'豆瓣读书'的工作表
myXlsxFile = xlsxwriter.Workbook('book_info.xlsx')
myWorkSheet = myXlsxFile.add_worksheet('豆游读书')

nums = len(titles)  # 根据标题数量获取记录数
# 第一行写入列名
myWorkSheet.write(0, 0, '图书标题')
myWorkSheet.write(0, 1, '图书作者')
myWorkSheet.write(0, 2, '出版社')
myWorkSheet.write(0, 3, '图书价格')
myWorkSheet.write(0, 4, '图书评分')
myWorkSheet.write(0, 5, '图书简介')
myWorkSheet.write(0, 6, '资源地址')
# 设置列宽
myWorkSheet.set_column('A:A', 20)
myWorkSheet.set_column('B:B', 20)
myWorkSheet.set_column('C:C', 30)
myWorkSheet.set_column('D:D', 20)
myWorkSheet.set_column('E:E', 10)
myWorkSheet.set_column('F:F', 100)
myWorkSheet.set_column('G:G', 50)
# 写入图书数据
for i in range(1, nums):
    myWorkSheet.write(i, 0, titles[i])
    myWorkSheet.write(i, 1, authors[i])
    myWorkSheet.write(i, 2, publishers[i])
    myWorkSheet.write(i, 3, prices[i])
    myWorkSheet.write(i, 4, ratings[i])
    myWorkSheet.write(i, 5, comments[i])
    myWorkSheet.write(i, 6, URLs[i])
myXlsxFile.close()  # 关闭电子表格
