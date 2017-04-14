# -*- coding=utf-8 -*-

import os
import xlwt
import codecs
import numpy as np
import pandas as pd
import jieba
import pickle
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


def delete_unimportant(words):
    result = []
    # 加再停词表
    ftm = open('data/stop_word.txt', 'r', encoding='utf-8')
    words_spilt = ' '
    for i in ftm.readlines():
        words_spilt += i
    ftm.close()
    for i in range(len(words)):
        if words[i] not in words_spilt:
            result.append(words[i])
    del words
    return result

# 创建字典
def create_dict(filename):
    neg_txt = pd.read_excel(filename, header=None, index_col=None)
    neg_txt['mark'] = 0
    neg_len = len(neg_txt)
    print('neg len : ', neg_len)
    del neg_txt[0]
    neg_txt['sentence'] = neg_txt[1]
    del neg_txt[1]
    # 分词函数
    cut_word = lambda sentence: np.asarray(delete_unimportant(list(jieba.cut(sentence))), dtype=str)
    # delete_unimport = lambda words:
    txt_list = np.asarray(neg_txt['sentence'], dtype=str)
    neg_word = pd.Series(map(cut_word, txt_list))
    neg_word = pd.DataFrame(neg_word, )
    neg_txt['words'] = neg_word
    # print(neg_txt['words'])
    return neg_txt

# 生成字典
# neg_txt = create_dict(xls_file_name)
# ftm = open('tmp/neg_txt.pkl', 'wb')
# pickle.dump(neg_txt, ftm)
# ftm.close()

ftm = open('tmp/neg_txt.pkl', 'rb')
neg_txt = pickle.load(ftm)
ftm.close()
# print(neg_txt['words'])
words_list = []
for i in neg_txt['words']:
    words_list.extend(i)
print(len(words_list))
dict_tmp = pd.DataFrame(pd.Series(words_list).value_counts())
dict_tmp['id'] = list(range(1, len(dict_tmp) + 1))
# print(dict_tmp['id'])
get_sent = lambda x: list(dict_tmp['id'][x])
neg_txt['sent'] = list(map(get_sent, neg_txt['words']))
print(neg_txt['sent'])
# 将txt转成xls
# txt2xls(files_path=dir_path, xls_name=xls_file_name, txt_encoding='gbk')
#clear_nan_repeat(xls_file_name)
# 两个xls合并
# df1 = pd.read_excel('', header=None, index_col=None)
# df2 = pd.read_excel('', header=None, index_col=None)
# df_sum = pd.concat([df1, df2])