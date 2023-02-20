# -*-coding:utf-8-*-
import re

str = '<p>你好</p>'
result = re.findall('<p>(.*?)</p>', str)
print(result)

str = """<p>你好
</p>"""
result = re.findall('<p>(.*?)</p>', str, re.S)
print(result)
print(result[0].strip())

