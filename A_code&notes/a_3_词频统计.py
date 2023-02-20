# -*-coding:utf-8-*-

with open('hamlet.txt', 'r', encoding='utf-8') as fr:
    text = fr.read().lower()
    # text = text.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')\
    #     .replace(';', ' ').replace(':', ' ')
    for char in "!'#$%&()*+,-./:;<=>?@[\\\\]^_{|}~":
        text = text.replace(char, ' ')
        # print(text)
text = text.split()
counts = {}
for i in text:
    counts[i] = counts.get(i, 0) + 1

items = list(counts.items())

for i in sorted(items, key=lambda x: x[1], reverse=True)[:10]:
    print('{:<6} -> {}'.format(i[0], i[1]))


