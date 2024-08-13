# -*-coding:utf-8-*-

import time

scale = 10
for i in range(scale + 1):
    a = '**' * i
    b = '..' * (scale - i)
    c = (i / scale) * 100
    # print('{:<3.0f} [{} -> {}]'.format(c, a, b))
    # time.sleep(0.1)

aa = 'hello world'
print(aa.index('w', 1, -1))

ls = [20, 10, 7, 6, 31]
print(sum(ls) / len(ls))

ll = [23, 45, 78, 87, 11, 67, 89, 13, 243, 56, 67, 311, 431, 111, 141]
ls = ll.copy()
count = 0
for i in ll:
    if i % 3 == 0:
        ls.remove(i)
        count += 1

print(ls, count)

shop_dict = {'卡布奇诺': 32, '摩卡': 30, '抹茶蛋糕': 28, '布朗尼蛋糕': 36}
print(sum(shop_dict.values()))
total = 0
for i in shop_dict:
    total += shop_dict.get(i)
print(total)
