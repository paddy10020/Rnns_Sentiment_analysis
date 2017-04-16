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

# 创建训练集
def create_corpus(filename, inclination):
    # inclination: 感情倾向：1 / 0（1是积极，0是消极）
    # example：
    # neg_txt = create_corpus(xls_file_name, 1)
    txt = pd.read_excel(filename, header=None, index_col=None)
    txt['mark'] = inclination
    txt_len = len(txt)
    print('neg len : ', txt_len)
    txt['sentence'] = txt[0]
    del txt[0]
    # 分词函数
    cut_word = lambda sentence: np.asarray(delete_unimportant(list(jieba.cut(sentence))), dtype=str)
    # delete_unimport = lambda words:
    txt_list = np.asarray(neg_txt['sentence'], dtype=str)
    txt_word = pd.Series(map(cut_word, txt_list))
    txt_word = pd.DataFrame(txt_word, )
    txt['words'] = txt_word
    # 保存
    txt_name = ''
    if inclination is 0:
        txt_name = 'neg_txt.pkl'
    else:
        txt_name = 'pos_txt.pkl'
    ftm = open('tmp/' + txt_name, 'wb')
    pickle.dump( txt, ftm)
    ftm.close()
    # print(neg_txt['words'])
    return txt

# 创建词典，每个词对应的id是训练集的中词的出现次数，,id是每个词的排名
def create_dict(txt_list):
    # example
    # dict_tmp = create_dict(neg_txt)
    words_list = []
    for i in txt_list['words']:
        words_list.extend(i)
    # print(len(words_list))
    dict_tmp = pd.DataFrame(pd.Series(words_list).value_counts())
    dict_tmp['id'] = list(range(1, len(dict_tmp) + 1))
    # 保存
    ftm = open('tmp/dick.pkl', 'wb')
    pickle.dump(dict_tmp, ftm)
    ftm.close()
    return dict_tmp

# 生成语句序列
def create_sequence(dict_tmp, train_data):
    get_sent = lambda x: list(dict_tmp['id'][x])
    train_data['sent'] = list(map(get_sent, train_data['words']))
    return train_data

ftm = open('tmp/neg_txt.pkl', 'rb')
neg_txt = pickle.load(ftm)
ftm.close()

dict_tmp = create_dict(neg_txt)

# print(dict_tmp)

neg_txt = create_sequence(dict_tmp, neg_txt)
print(neg_txt['sent'])

# 将txt转成xls
# txt2xls(files_path=dir_path, xls_name=xls_file_name, txt_encoding='gbk')
#clear_nan_repeat(xls_file_name)
# 两个xls合并
# df1 = pd.read_excel('', header=None, index_col=None)
# df2 = pd.read_excel('', header=None, index_col=None)
# df_sum = pd.concat([df1, df2])


