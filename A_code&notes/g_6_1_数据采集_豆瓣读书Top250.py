# -*-coding:utf-8-*-

# 导入相应的库文件
from lxml import etree
from fake_useragent import UserAgent
import requests
import csv

# 创建csv
fp = open('result.csv', 'wt', newline='', encoding='utf-8')
writer = csv.writer(fp)
writer.writerow(('name', 'URL', 'author', 'publisher', 'date', 'price', 'date', 'price', 'rate', 'comment'))
# 构造URL
URLs = ['https://book.douban.com/top250?start={}'.format(str(i)) for i in range(0, 250, 25)]

headers = {
    'User-Agent': UserAgent().random
}
# "User - Agent':Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36(KHTMI，like Gecko) Chrome/53.0.2785143 Safari/537.36
for URL in URLs:
    html = requests.get(URL, headers=headers)
    selector = etree.HTML(html.text)
    infos = selector.xpath('//tr[@class ="item"]')
    for info in infos:
        name = info.xpath('td/div/a/@title')[0]
        URL = info.xpath('td/div/a/@href')[0]
        book_infos = info.xpath('td/p/text()')[0]
        author = book_infos.split('/')[0]
        publisher = book_infos.split('/')[-3]
        date = book_infos.split('/')[-2]
        price = book_infos.split('/')[-1]
        rate = info.xpath('td/div/span[2]/text()')[0]
        comments = info.xpath('td/p/span/text()')
        comment = comments[0] if len(comments) != 0 else "空"
        writer.writerow((name, URL, author, publisher, date, price, date, price, rate, comment))
fp.close()
