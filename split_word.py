# -*- coding=utf-8 -*-

import jieba
import numpy as np
import codecs
import xlrd

filename = 'sum'

# 将xls文档转为txt文档
# data_workbook = xlrd.open_workbook('pos' + '.xls')
#
# table = data_workbook.sheets()[0]
#
# text = table.col_values(0)
# f1 =codecs.open('pos'+'.txt', 'w', 'utf-8')
# for i in text:
#     f1.write(i + '\n')
# f1.close()

# 分词

f1 = codecs.open(filename+'.txt', 'r', 'utf-8')
f2 = codecs.open(filename+'.fen.txt', 'w', 'utf-8')
for i in f1.readlines():
    seg_list = jieba.cut(i)
    tmp = ' '.join(seg_list)
    f2.write(tmp + '\n')
f1.close()
f2.close()