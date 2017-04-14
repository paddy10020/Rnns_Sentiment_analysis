# -*- coding=utf-8 -*-
import codecs

# 把txt合起来
filename = []
text_list = []
file_sum = codecs.open('sum.txt', 'w', 'utf-8')
while True:
    input_file = input('Please input filename')
    if not input_file == '':
        file_temp = codecs.open(input_file, 'r', 'utf-8')
        for i in file_temp.readlines():
            file_sum.write(i + '\n')
        file_temp.close()

    else:
        break

file_sum.close()

