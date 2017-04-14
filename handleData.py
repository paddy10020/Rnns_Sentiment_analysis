# -*- coding=utf-8 -*-

import os
import xlwt
import codecs
import numpy as np
import pandas as pd
from pandas import Series


def txt2xls(dir_path, xls_file_name, txt_encoding='gbk'):
    """
    将文件夹中的所有文件都导入到xls中
    """
    txt_list = []
    file_count = 0
    file_name_list = os.listdir(dir_path)
    for file_name in file_name_list:
        file_count += 1
        try:
            file_tmp = codecs.open(dir_path + file_name, 'r', encoding=txt_encoding)
            for txt_tmp in file_tmp.readlines():
                if txt_tmp is not '' or None:
                    txt_list.append(txt_tmp.strip())
        except Exception as e:
            print('open file error. The file name is ', file_name)
            print(e)
            if file_count >= len(file_name_list):
                break
    txt_count = len(txt_list)
    print('The count of txt is ', txt_count)
    txt_list = np.asarray(txt_list)
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('0')
    for i in range(txt_count):
        worksheet.write(i, 0, txt_list[i])
    workbook.save(xls_file_name)

dir_path = 'data/neg/'
xls_file_name = 'data/neg.xls'
# txt2xls(dir_path=dir_path, xls_file_name=xls_file_name, txt_encoding='gbk')
xls_file = pd.read_excel(xls_file_name, header=None, index_col=None)
print(xls_file.duplicated())
xls2 = xls_file.drop_duplicates()
print(xls2.duplicated())
# 数据去重复和去空


# while True:
#     try:
#         with codecs.open(txt_file_path+str(file_count)+'.txt', 'r', encoding='gbk') as file:
#             a = file.read().strip('\n')
#             txt_array.append(a)
#             # print(a)
#             file.close()
#         count += 1
#         file_count += 1
#     except:
#         file_count += 1
#         if count >= 4000:
#             break
#
# txt_array = np.asarray(txt_array)
# workbook = xlwt.Workbook(encoding = 'utf-8')
# worksheet = workbook.add_sheet('0')
# for i in range(len(txt_array)):
#     worksheet.write(i, 0, txt_array[i])
# # worksheet.write(0, 0, 'Row 0, Column 0 Value')
# # worksheet.write(1,0, 'Row 1, Column 0 Value')
# workbook.save(xlsName)
