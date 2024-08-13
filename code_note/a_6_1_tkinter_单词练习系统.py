# -*-coding:utf-8-*-
import tkinter
import math
import tkinter.messagebox
import random
import time
from tkinter import ttk


class Recite():

    def __init__(self):
        """
        初始化设置：创建系统类，在类中创建构造方法，设置各种初始化变量。
        """
        # 创建tkinter窗口对象，设置窗口属性。
        self.root = tkinter.Tk()
        self.root.geometry('450x500')
        self.root.title('背单词')
        self.root['bg'] = 'yellow'
        # 创建显示单词变量
        self.word = tkinter.StringVar()
        self.score = tkinter.StringVar()
        self.fen = 0
        self.score.set(0)
        self.prompt = tkinter.StringVar()  # 提示信息
        self.prompt.set('你最棒')

        self.wrong = []                             # 创建错词列表变量
        self.dic = []                               # 创建单词列表变量
        self.word_list()
        # print(len(self.dic))
        self.radiolist = tkinter.IntVar()
        self.fill = ''                              # 创建拼写填空的变量
        self.space = ''
        # 创建随机数变量（用于从单词表中随机抽取单词）
        self.r = random.randint(0, len(self.dic) - 1)
        self.layout()
        self.root.mainloop()

    def word_list(self):
        """
        读取数据，先对数据进行清洗，按换行符将它们分离成列表，一个单词为一个元素。
        再对列表进行遍历，对于遍历到的每一个元素即单词再次进行分离，即每个单词分
        成中文和英文两个元素，将分离得到的列表追加到初始化设置中创建的单词列表变量中
        :return:
        """
        f = open('words.txt', 'r', encoding='utf-8')
        t = f.read().split('\n')

        for d in t:
            self.dic.append(d.split())
        # print(self.dic)
        f.close()

    def layout(self):
        """
        主界面设计
        :return:
        """
        # 标题：为标签，放置到顶部
        lab1 = tkinter.Label(self.root, text='背单词，赢积分',
                             font=('宋体', 30), bg='yellow')
        lab1.pack(pady=20)
        # 积分：标签+标签，ᨀ示标签中的文本属性为“当前积分”，积分标签中的内容为文本变量，显示初始设置中的积分变量
        lab_score = tkinter.Label(self.root, textvariable=self.score,
                                  font=('宋体', 30), fg='red', bg='yellow')
        lab_score.pack()
        lab_2 = tkinter.Label(self.root, text='当前积分：',
                              font=('宋体', 16), bg='yellow')
        lab_2.place(x=40, y=100)

        # 答案输入：输入文本框。
        self.entry = tkinter.Entry(self.root, width=15, font=('宋体', 20))
        self.entry.place(x=160, y=140)

        # 单词显示：标签，文本为初始设置中的显示单词变量。
        lab_word = tkinter.Label(self.root, textvariable=self.word,
                                 font=('宋体', 20), bg='white')
        lab_word.place(x=160, y=180)
        # 赞扬提示和鼓励提示语位置
        lab_prompt = tkinter.Label(self.root, textvariable=self.prompt, font=('宋体', 18), fg='blue', bg='yellow')
        lab_prompt.place(x=140, y=250)

        r1 = tkinter.Radiobutton(self.root, variable=self.radiolist,
                                 value=0, text='英译中', command=self.select1, bg='yellow')
        r2 = tkinter.Radiobutton(self.root, variable=self.radiolist,
                                 value=1, text='中译英', command=self.select2, bg='yellow')
        r3 = tkinter.Radiobutton(self.root, variable=self.radiolist,
                                 value=2, text='拼写填空', command=self.select3, bg='yellow')
        # 回答提示：标签，文本为初始设置中的提示变量。
        self.radiolist.set(0)
        r1.place(x=40, y=130)
        r2.place(x=40, y=150)
        r3.place(x=40, y=170)
        # 判断（确定）：按钮，通过command属性调用判断方法。
        but1 = tkinter.Button(self.root, text='确定', width=5, font=('宋体', 15), command=self.judge)
        but1.place(x=130, y=300)
        # 退出：按钮，通过command属性调用退出方法。
        but2 = tkinter.Button(self.root, text='退出', width=5, font=('宋体', 15), command=self.exit)
        but2.place(x=230, y=300)
        # 查看错词表：按钮，通过command属性调用错词显示方法。
        but3 = tkinter.Button(self.root, text='查看错词表', width=10, font=('宋体', 15), command=self.wrong_word)
        but3.place(x=130, y=400)

    def select1(self):
        """
        单选“英译中”：标签显示英文，随机显示单词表中的一个单词。
        """
        self.r = random.randint(0, len(self.dic) - 1)
        self.word.set(self.dic[self.r][0])

    def select2(self):
        """
        单选“中译英”：随机显示单词表中一个词语，标签显示中文。
        """
        self.r = random.randint(0, len(self.dic) - 1)
        self.word.set(self.dic[self.r][1])

    def select3(self):
        """单选“拼写填空”：
        单词字母随机缺少一个。利用随机数生成一个数字，遍历英文单词的字母，数字应对的位置替换为下画线，其他的字母正常显示。
        """
        self.r = random.randint(0, len(self.dic) - 1)
        word = self.dic[self.r][0]
        k = random.randint(0, len(word) - 1)
        self.space = ''
        for i in range(len(word)):
            if i != k:
                self.space += word[i]
            else:
                self.space += '_'
                self.fill = word[i]
        self.space = self.space + '' + self.dic[self.r][1]
        self.word.set(self.space)

    def judge(self):
        """
        正误判断设计
        先判断练习方式是哪一种，再获取文本框输入的内容，将获取的内容与对应的单词表内容进行比对，
        如果相同，则增加积分，给出赞扬提示；如果不同，则给出鼓励ᨀ示，并将对应单词加到错词表。
        :return:
        """
        # 拼写填空模式
        if self.radiolist.get() == 2:
            # 统一变成小写
            s = self.entry.get().lower()
            # 输入内容与空缺内容相同，则给出赞扬ᨀ示，增加积分，将积分变量进行更新并显示在标签中。
            if s == self.fill:
                self.prompt.set('太棒了')
                self.fen += 1
                self.score.set(self.fen)
            # 输入内容与空缺内容不同，则给出鼓励提示，将单词追加到错词列表，
            else:
                self.prompt.set('很遗憾，继续加油')
                self.wrong.append(self.dic[self.r])
            # 再次生成随机数，用于下一轮的单词抽取，调用拼写填空方法，并将文本输入框的内容清空
            self.r = random.randint(0, len(self.dic) - 1)
            self.select3()
            self.entry.delete(0, 'end')
        # 中译英 或者 英译中
        else:
            # 中译英模式
            if self.radiolist.get() == 0:
                e = 0
                c = 1  # c=1，即与文本框内容匹配的是列表中的索引号为1，提取词语中的中文
            # 英译中模式
            elif self.radiolist.get() == 1:
                e = 1
                c = 0  # c=0，即与文本框内容匹配的是列表中的索引号为0，提取单词中的英文。
            word = self.dic[self.r][c]
            s = self.entry.get()
            if word == s:
                # 将文本框内容与ᨀ取到的内容进行比对，如果相同，则给出赞扬ᨀ示，增加积分，更新积分并显示
                self.prompt.set('太棒了')
                self.fen += 1
                self.score.set(self.fen)
            else:
                # 如果不同，则给出鼓励ᨀ示，将单词追加到错词表。
                self.prompt.set('很遗憾，继续加油')
                self.wrong.append(self.dic[self.r])
            # 生成随机数，用于下一轮单词抽取，将抽取到的单词更新并显示出来，清空文本框中的内容
            self.r = random.randint(0, len(self.dic) - 1)
            self.word.set(self.dic[self.r][e])
            self.entry.delete(0, 'end')

    def exit(self):
        """退出程序"""
        # 单击“退出”按钮，退出程序
        self.root.destroy()

    def wrong_word(self):
        """
        错词表设计
        单击“查看错词表”按钮调用错词方法，打开新的窗口，原窗口隐藏。
        在新窗口将错词表中的内容以表格的形式列出来，并显示错误次数。
        """
        # 隐藏原窗口，创建新窗口，显示标题：标签
        self.root.withdraw()
        self.wt = tkinter.Tk()
        self.wt.title('错词表')
        self.wt.geometry('450x500')
        self.wt['bg'] = 'yellow'
        lab_wr = tkinter.Label(self.wt, text='本次练习错词表', font=('宋体', 20), bg='yellow')
        lab_wr.place(x=100, y=10)
        # 创建表格：表格位置、列数目3，列标题
        tree = ttk.Treeview(self.wt, show='headings', height=15)
        tree.place(x=30, y=50)
        tree['columns'] = ('1', '2', '3')
        tree.column('1', width=110)
        tree.column('2', width=110)
        tree.column('3', width=110)

        tree.heading('1', text='英文', anchor='center')
        tree.heading('2', text='中文', anchor='center')
        tree.heading('3', text='错误次数', anchor='center')

        but_re = tkinter.Button(self.wt, text='返回', font=('宋体', 15), command=self.back)
        but_re.place(x=50, y=400)
        # 对错词表进行排序，让相同的单词相邻。
        self.wrong.sort(key=lambda x: x[0])
        # 在错词表末尾追加一个符号，用于遍历结束，便于统计次数。
        self.wrong.append(['', ''])
        p = self.wrong[0][0]
        c = 1
        # 获取错词中的第一个单词，并设出现次数为1，从第二个单词开始对单词表进行遍历
        for w in range(1, len(self.wrong)):
            # 如果单词与第一个单词相同，则将次数加1。
            if self.wrong[w][0] == p:
                c += 1
            # 如果不同，则将单词的英文、中文、次数写入表格中并显示，同时将次数修改为1，第一个单词的变量值改为当前单词。
            else:
                tree.insert("", 'end', values=(self.wrong[w - 1][0], self.wrong[w - 1][1], c))
                # print('{}\t{}\t{}'.format(self.wrong[w][0], self.wrong[w][1], c))
                c = 1
                p = self.wrong[w][0]
        # 将添加的符号删除，以防止下次对它进行统计
        self.wt.protocol('WM_DELETE_WINDOW', self.back)

    def back(self):
        """返回设计"""
        self.wt.destroy()           # 关闭（销毁前一个）destroy（）
        self.root.update()          # 更新原隐藏主页面update()
        self.root.deiconify()       # 显示原隐藏主页面deiconify()

if __name__ == '__main__':
    word_system = Recite()
    word_system.layout()

