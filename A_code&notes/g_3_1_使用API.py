# -*-coding:utf-8-*-
import requests

params = {
    'query': '蜈支洲岛',
    'region': '三亚',
    'output': 'json',
    'ak': 'X53ztMqptZuBf621xoyvI7t7RlXkOB2i'
}
URL = 'https://api.map.baidu.com/place/v2/search'
res = requests.get(URL, params)
print(res.text)
