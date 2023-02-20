# -*-codiyg:utf-8-*-

def run_year_judge(y):
    """
    现闰年的判断。
    当年份能被4整除且不能被100整除时、年份能被400整除时为闰年，其余为平年。
    定义函数的返回值：为闰年则返回True，为平年则返回False。
    :param y:
    :return:
    """
    if y % 4 == 0 and y % 100 != 0 or y % 400 == 0:
        return True
    else:
        return False


def month_day_num(y, m):
    """
    获取每个月的天数。
    当为闰年时，2月为29天，而当为平年时，2月为28天。其他月份中：1、3、5、7、8、10、12为31天，4、6、9、11为30天。
    :param y:
    :param m:
    :return:
    """
    if m == 2:
        if run_year_judge(y) is True:
            return 29
        else:
            return 28
    elif m in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    else:
        return 30


def days_judge(y, m):
    """
    1990年，当前年的1月1日为星期一
    获取指定年、月到参考年、月的总天数，以确定每个月的1号是星期几。
    设定天数初始值为0，先判断当年是否为闰年，如果为闰年，则天数加366天，
    如果为平年，则天数加365天。再加上当前年从1月到指定月份的天数。
    :param y:
    :param m:
    :return:
    """
    days = 0
    for i in range(1990, y, 1):
        if run_year_judge(i) is True:
            days += 366
        else:
            days += 365
    for i in range(1, m, 1):
        days += month_day_num(y, i)
    return days


def calendar(y, m):
    """
    将日期按星期进行排列，一周为七天，需定义一个计数器，初始值为0，用来控制输出换行，当数值能被7整除时则换行。
    :param y:
    :param m:
    :return:
    """
    print('\t\t{}年{}月份日历'.format(y, m))
    print('Sun Mon Tues Wed Thur Fri Sat')
    print('-' * len('Sun Mon Tues Wed Thur Fri Sat'))
    count = 0
    # 当前月份的1号是星期几，将前面的星期位置空出来
    for i in range(1, (days_judge(y, m) + 1) % 7):
        # 确定日历的输出顺序，必须先确定1号的星期位置
        # 星期一排在第一个，则直接用总天数对7整除取余数
        # 星期日排在第一个，则将总天数加1再对7整除取余数
        # 1号之前的星期位置需要空出来，所以直接输出间隔符，每输出一个间隔符，都需要对计数器加1
        print(end='\t ')
        count += 1
    # 按星期位置输出每个月的天数
    for i in range(1, month_day_num(y, m) + 1):
        # 从1号按顺序输出日期，每输出一个日期，计数器就加1，当计数
        # 器的值能被7整除时，则换行
        print(i, end='\t ')
        count += 1
        if count % 7 == 0:
            print()


if __name__ == '__main__':
    calendar(2022, 12)
    # count = 0
    # print(days_judge(2022, 12) % 7)
    # for i in range(1, days_judge(2022, 12) % 7 + 1):
    #     count += 1
    #     print(end='\t')
    #     count += 1
    # print(count)
    # for i in range(1, month_day_num(2022, 12) + 1):
    #     print(i, end='\t')
    #     count += 1
    #     if count % 7 == 0:
    #         print()
