# -*-coding:utf-8-*-

aa = "a = 1, b = 2, c = 3"
# print(dict(a.split('=', a) for a in aa.split('=')))
print(dict((l.split('=')) for l in aa.split(',')))
print(dict(((lambda i: (i[0], int(i[1])))(l.split('=')) for l in aa.split(','))))
print(dict(a=1, b=2, c='qq'))
print(dict(((lambda i: (i[0], int(i[1])))(l.split('=')) for l in aa.split(','))))
print([(lambda i: (i[0], int(i[1])))(l.split('=')) for l in aa.split(',')])
