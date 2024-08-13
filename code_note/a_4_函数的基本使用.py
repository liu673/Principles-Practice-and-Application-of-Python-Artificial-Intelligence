# -*-coding:utf-8-*-

def fact(n):
    s = 1
    for i in range(1, n+1):
        s *= i
    return s

def jiecheng(n, m):
    fenzi = fact(n)
    fenmu = fact(m) * (n - m)
    return fenzi / fenmu

c = jiecheng(3, 2)
print(c)

