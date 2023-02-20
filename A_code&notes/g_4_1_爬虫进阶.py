# -*-coding:utf-8-*-
"""
（1）动态UA
（2）代理IP
（3）模拟登陆
（4）验证码问题
（5）网页动态内容的获取
（6）多线程与多进程（多进程爬虫使用了multiprocessing库）
"""

# 动态UA
"""
fake_useragent库可以方便地获取多种浏览器的UA，具体使用的时候直接加入headers字典里就可以了
"""
import requests
from fake_useragent import UserAgent

# ua = UserAgent()  # 得到UA对象
# print(ua.edge)  # EDGE浏览器的User Agent
# print(ua.Chrome)
# print(ua.firefox)
# print(ua.ie)
# print(ua.opera)
# print(ua.safari)
# print(ua.random)  # 支持随机生成请求头

# 代理IP
"""
对于封IP的网站，Requests库ᨀ供了方便的方法来使用代理IP
"""
proxies = {  # 构造存储代理IP地址的字典
    'http': '218.71.161.56:80',
    # 'http': '19.85.5.75:24748',
    # 'http': '222.85.39.52:808'
}
# data = requests.get('http://sanya.xiaozhu.com/', proxies=proxies)
# print(data.text)

# 模拟登陆
"""
有些网站需要登录账号才能看到一些数据，想要抓取这些数据，就必须先登录。而浏览器主要通过cookie的方式来检验用户的登录状
态，因此可以直接通过从浏览器复制cookie到headers进行模拟登录。cookie的获取方式和之前的UA的获取方式是一样的，
"""
mycokie = ''
ua = UserAgent()
headers = {
    'UserAgent': ua.random,
    'Cookies': mycokie
}
URL = 'https://www.douban.com/people/222710992/'
# data = requests.get(URL, headers=headers)
# print(data.status_code)
# print(data.request.headers)
# print(data.text)

# 验证码问题
"""
第一种就是ᨀ取验证码的地址，下载验证码到本地手动输入后再使用post登录。这种方式需要人工参与，操作较为烦琐。第
第二种是通过一些验证码识别的库，如pytesseract库，进行识别，但是遇到复杂的验证码识别率就会很低。
第三种是使用云打码平台。此方法识别率高，不需要人工操作，缺点是需要收费。
"""
from PIL import Image
import pytesseract

image = Image.open('img.png')  # 验证码图片
# content = pytesseract.image_to_string(image)  # 解析图片
# print(content)

# 网页动态内容的获取
"""
第一种是直接从网页响应中找到JS脚本返回的JSON数据，其难点在于包含数据文件地址的查找，优点是针对性较强，速度快；
第二种方法是使用Selenium对网页进行模拟访问，其缺点是处理速度较慢，优点是简单易用。
"""
import pprint


def getdata(json_URL):
    ua = UserAgent()
    headers = {'User-Agent': ua.random}
    data = requests.get(json_URL, headers=headers)
    pprint.pprint(data.text)


# json_URL = ''
# getdata(json_URL)

# 多线程与多进程
"""
串行下载极大地限制了爬虫的速度和效率，尤其不适用于大批量的请求处理。
多线程即请求任务同时进行，程序的执行在不同的线程之间进行切换，每个线程执行程序的不同部分。多进程的原理与多线
程的原理比较类似，多进程就是在多核CPU的不同核上进行进程的切换执行

从输出的结果可以看出4进程并行的爬虫速度快于2进程爬虫并行的速度，2进程并行爬取的速度又快于串行爬取的速度。这就是使用多进程和多线程的意义。
"""
import requests
import re
import time
from fake_useragent import UserAgent
from multiprocessing import Pool  # 添加多进程multiprocessing库

ua = UserAgent()
headers = {'User-Agent': ua.random}  # 使用动态UA


def re_scraper(URL):
    res = requests.get(URL, headers=headers)
    ids = re.findall('<h2>(.*?)</h2>', res.text, re.S)
    contents = re.findall('<div class = "content">.*?<sapn>(.*?)</span>', res.text, re.S)
    laughs = re.findall('<span class="stats-vote"><i class = "number>(\d+)</i></span>', res.text, re.S)
    comments = re.findall('<i class = "number">(\d+)</i>评论', res.text, re.S)
    for id, content, laugh, comment in zip(ids, contents, laughs, comments):
        info = {
            'id': id,
            'content': content,
            'laugh': laugh,
            'comment': comment
        }
        return info


# if __name__ == '__main__':
#     URLS = ['http://www.qiushibaike.com/text/page/{}/'.format(str(i)) for i in range(1, 36)]
#     # 串行进程
#     start_1 = time.time()
#     for URL in URLS:
#         re_scraper(URL)
#     end_1 = time.time()
#     print('串行爬虫', end_1 - start_1)
#     # 2个进程串行
#     start_2 = time.time()
#     pool = Pool(processes=2)
#     pool.map(re_scraper, URLS)
#     end_2 = time.time()
#     print('2个进程', end_2 - start_2)
#     # 4个进程串行
#     start_3 = time.time()
#     pool = Pool(processes=4)
#     pool.map(re_scraper, URLS)
#     end_3 = time.time()
#     print('4个进程', end_3 - start_3)
#     # 串行爬虫 138.39173078536987
#     # 2个进程 80.09881901741028
#     # 4个进程 37.76795673370361
