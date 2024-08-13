# -*-coding:utf-8-*-

""""
网络爬虫是在万维网浏览网页并按照一定规则ᨀ取信息的脚本或者程序。用户在浏览网页时，浏览器向Web服务器发出请求，在浏览器
中展示选择的网络资源，资源一般为HTML文档，资源的位置由用户使用URL（统一资源定位符）指定
用脚本模仿浏览器，向网站服务器发出浏览网页内容的请求，在服务器检验成功后，返回网页的信息，然后解析网页并提取需要的数据，最后将提取的数据保存即可。
步骤：
（1）使用Requests库发起请求
（2）服务器检验请求：大量的爬虫请求会造成服务器压力过大，可能使得网页响应速度变慢，影响网站的正常运行。所以网站一般会检验请求头里边的User-Agent（以下简称UA，相当于身份的识别）来判断发起请求的是不是机器人，而我们可以通过自己设置UA进行简单伪装。
（3）解析网页并提取数据：BeautifulSoup库和正则表达式
（4）保存提取的内容

注意：有些网站设置robots.txt声明对爬虫的限制     http://www.robotstxt.org/
"""
import requests
from bs4 import BeautifulSoup

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0;Win64; x86) AppleWebKit/537.36 (KHTML, like Gecko)'
                         'Chrome/85.0.4183.83 Safari/537.36'}
URL = 'https://sanya.xiaozhu.com/'
res = requests.get(URL, headers=headers)
"""
Requests库不仅有get（）方法，还有post（）方法，post（）方法用于ᨀ交表单来爬取需要登录才能获得数据的网站，
print(res.text)
所 有 Requests 显 示 抛 出 的 异 常 都 继 承 自requests.exceptions.RequestException，当发现这些错误或异常时，需要进行代码修改然后再重新运行代码,
爬虫程序重新运行，爬取到的数据又会重新爬取一次，这对于爬虫的效率和质量来说都是不利的。这时，可以通过Python中的try来避免异常，
"""
# try:
#     print(res.text)
# except ConnectionError:
#     print('拒绝链接')
soup = BeautifulSoup(res.text, 'html.parser')
# print(soup.prettify())
"""
BeautifulSoup库官方推荐使用lxml作为解析器，因而效率更高。
find(tag,attributes,recursive,text,keywords)
find_all(tag,attributes,recursive,text,limit,keywords
#查找div标签,class=＂item＂
 soup.find_all('div',＂item＂)
 soup.find_all('div',class='item')
 soup.find_all('div',attrs={＂class＂:＂item＂})
 
find_all（）方 法 返 回 文 档 中 所 有 符 合 条 件 的 tag 的 集 合（ class'bs4.element.ResultSet' ） ， 
而 find （ ） 方 法 返 回 一 个tag（class'bs4.element.Tag'）

select（）方法可以根据网页内标签选择器的位置关系进行筛选，
soup.select(div.item>a>h1)
"""

"""
Lxml
使用C语言编写，解析速度比BeautifulSoup更快，可以很好地支持HTML文档的解析功能，也可以从HTML文件中ᨀ取内容。

XML库使用Xpath语法解析并定位网页数据。Xpath语言具有在XML文档中查找信息的作用。根据XML文件中的节点（标签）关系，Xpath使用路径表达式在XML文档中选取节点。
"""
from lxml import etree

html = etree.HTML(res.text)
# result = etree.tostring(html)
# print(result)
selector = etree.HTML(res.text)
id = selector.xpath('//*........')












