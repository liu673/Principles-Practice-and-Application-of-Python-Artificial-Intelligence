# -*-coding:utf-8-*-

"""
https://blog.51cto.com/u_13567403/5018248
通过把字母移动一定的位数来实现加解密

明文中的所有字母从字母表向后（或向前）按照一个固定步长进行偏移后被替换成密文
"""
# ord() ：将字符转换为了对应的 ASCII 值
# chr()： 将对应的值转换为字符
text = 'Live is short,I use Python'
def KaiSa(text, key):
    word_list = []
    for p in text:
        p = p.lower()
        if 'a' <= p <= 'z':
            new_text = chr((ord(p) - ord('a') + int(key)) % 26 + ord('a'))
        else:
            new_text = p
        word_list.append(new_text)
    return "".join(word_list)

a = KaiSa(text, 3)
print(a)
tt = "olyh lv vkruw,l xvh sbwkrq"
# 解密 将移动步长换为负数
b = KaiSa(tt, -3)
print(b)














