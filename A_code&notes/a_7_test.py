# -*-coding:utf-8-*-
import random


def isprime(number):
    """
    参数为整数，并且需要有异常处理功能。
    此函数的功能是检测接收的整数是否为质数，如果整数是质数，则返回True，否则返回False
    """
    try:
        if type(number) == int:
            pass
        if number > 1:
            for i in range(2, number):
                if number % i == 0:
                    print(i, "乘以", number // i)
                    return False
                else:
                    return True
    except:
        return '请检查此数是否为整数'


class gongyueshu_gongbeishu():
    """最大公约数和最小公倍数计算"""
    def __init__(self, number1: int, number2: int):
        self.number1 = number1
        self.number2 = number2

    def gongyueshu(self):
        a = [i for i in range(1, self.number1 + 1) if self.number1 % i == 0]
        b = [i for i in range(1, self.number2 + 1) if self.number2 % i == 0]

        if len(a) < len(b):
            return [i for i in a if i in b][-1]
        else:
            return [i for i in b if i in a][-1]

    def gongbeishu(self):
        return int(self.number1 * self.number2 / self.gongyueshu())


# a = gongyueshu_gongbeishu(12, 8)
# print(a.gongyueshu())
# print(a.gongbeishu())

times = 0
def hanoi(A, B, C, n):
    """
    递归函数
    汉诺塔问题是一个古典的数学问题，它只能用递归方法来解决。在古代有一个梵塔，塔内有A、B、C三个座。
    开始时A座上有64个盘子，盘子大小不同，但保证大的在下，小的在上。
    现在有一个和尚想将这64个盘子从A座移动到C座，但他每次只能移动一个盘子，且在移动过程中在3个座上都必须保持大盘在下小盘在上的状态。
    在移动过程中可以利用B座，要求编程将移动步骤打印出来。
    """
    global times           # 初始化次数
    # 若A座上只有1个盘子，此时N=1，则可直接将盘子从A座移动到C座上
    if n == 1:
        print(A, '->', C)
        times += 1
    # 若A座上有1个以上的盘子，即n > 1, 将n - 1个盘子视为整体
    else:
        # 先将N-1个盘子从A座借助C座移动到B座上。显然，这N-1个盘子不能作为一个整体移动，而是要按照要求来移动。可递归调用函数hanio(A,C,B,n-1)。
        # 这里借助C座将N-1个盘子从A座移动到B座，A是源，B是目标。
        hanoi(A, C, B, n - 1)
        # 将A座上剩下的第N个盘子（即最大的盘子）移动到C座上
        print(A, '->', C)
        times += 1
        # 将B座上的N-1个盘子借助A座移动到C座上。此时，递归调用函数hanio(B,A,C, n-1)
        # 这里借助于A座将N-1个盘子从B座移动到C座，B是源，C是目标。
        hanoi(B, A, C, n - 1)


# hanoi('A', 'B', 'C', 4)
# print(times)

import string
def random_password():
    """随机密码生成"""
    # str_list = list(string.ascii_lowercase) + list(string.ascii_uppercase)
    str_list = list(string.ascii_letters)
    number_list = list(str(i) for i in range(0, 11))
    password_list = str_list + number_list
    # a = random.choices(password_list, k=8)
    # n = "".join(random.sample(password_list, k=8))
    return ["".join(random.sample(password_list, k=8)) for i in range(10)]


print(random_password())



















