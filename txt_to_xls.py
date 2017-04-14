# -*- coding=utf-8 -*-

import numpy as np
import xlwt
import codecs
np.random.seed(1337)
txt_file_path = r'neg/'
xlsName = 'data/neg_data.xls'
file_count = 0
count = 0
# txt_count = 1949
txt_array = list()
# file = open(txt_file_path+str(file_count)+'.txt', 'r', encoding='gbk')
# a = file.read()
# print(a)
# file.close()

while True:
    try:
        with codecs.open(txt_file_path+str(file_count)+'.txt', 'r', encoding='gbk') as file:
            a = file.read().strip('\n')
            txt_array.append(a)
            # print(a)
            file.close()
        count += 1
        file_count += 1
    except:
        file_count += 1
        if count >= 4000:
            break

txt_array = np.asarray(txt_array)
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('0')
for i in range(len(txt_array)):
    worksheet.write(i, 0, txt_array[i])
# worksheet.write(0, 0, 'Row 0, Column 0 Value')
# worksheet.write(1,0, 'Row 1, Column 0 Value')
workbook.save(xlsName)

