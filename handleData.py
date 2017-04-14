# -*- coding=utf-8 -*-

import os
import xlwt
import codecs
import numpy as np
import pandas as pd
import jieba
from tempfile import NamedTemporaryFile
from pandas import Series


def txt2xls(files_path, xls_name, txt_encoding='gbk'):
    """
    将文件夹中的所有文件都导入到xls中
    """
    txt_list = []
    file_count = 0
    file_name_list = os.listdir(files_path)
    for file_name in file_name_list:
        file_count += 1
        try:
            file_tmp = codecs.open(files_path + file_name, 'r', encoding=txt_encoding)
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
    workbook.save(xls_name)


def clear_nan_repeat(file_name):
    """数据去重复和去空"""
    xls_file = pd.read_excel(file_name, header=None, index_col=None)
    print(xls_file.duplicated())
    xls2 = xls_file.drop_duplicates()
    xls2 = xls2.dropna()
    print(xls2.duplicated())
    xls2.to_excel(file_name, sheet_name='0')

dir_path = 'data/neg/'
xls_file_name = 'data/neg.xls'


# 将txt转成xls
# txt2xls(files_path=dir_path, xls_name=xls_file_name, txt_encoding='gbk')
#clear_nan_repeat(xls_file_name)
# 两个xls合并
# df1 = pd.read_excel('', header=None, index_col=None)
# df2 = pd.read_excel('', header=None, index_col=None)
# df_sum = pd.concat([df1, df2])

# 生成字典
neg_txt = pd.read_excel('data/neg.xls', header=None, index_col=None)
neg_txt['mark'] = 0
neg_len = len(neg_txt)
print('neg len : ', neg_len)
del neg_txt[0]
neg_txt['sentence'] = neg_txt[1]
del neg_txt[1]
# cut_word = lambda x: list(jieba.cutneg_txt['sentence'](x))
# neg_txt['word'] = np.asarray(neg_txt['sentence'], dtype=str).apply(cut_word)
txt_list = np.asarray(neg_txt['sentence'], dtype=str)

# print(neg_txt.head())
