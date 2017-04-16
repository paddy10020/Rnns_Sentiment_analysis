# -*- coding=utf-8 -*-

import os
import xlwt
import codecs
import numpy as np
import pandas as pd
import jieba
import pickle
from keras.utils import np_utils
from keras.preprocessing import sequence


from tempfile import NamedTemporaryFile

# 删除Cache
def deleteCache(fileName):
    if os.path.exists(fileName):
        os.remove(fileName)
    else:
        print('没有', fileName, '该文件')

# 保存Cache
def saveCache(fileName, tmp):
    fileTmp = open(fileName, 'wb')
    pickle.dump(tmp, fileTmp)
    fileTmp.close()


# 加载Cache
def loadCache(fileName):
    fileTmp = open(fileName, 'rb')
    tmp = pickle.load(fileTmp)
    fileTmp.close()
    return tmp

# 得到的数据缓存都会放大在tmp里面

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
    df = pd.DataFrame(xls_file)
    df[0] = df.drop_duplicates()
    # df.to_excel('data/test.xls', header=None, index=False)
    df.to_excel(file_name, header=None, index=False)

# 删除符号和停词
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
    print('txt len : ', txt_len)
    txt['sentence'] = txt[0]
    del txt[0]
    # 分词函数
    cut_word = lambda sentence: np.asarray(delete_unimportant(list(jieba.cut(sentence))), dtype=str)
    txt_list = np.asarray(txt['sentence'], dtype=str)
    txt_word = pd.Series(map(cut_word, txt_list))
    txt_word = pd.DataFrame(txt_word, )
    txt['words'] = txt_word
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
    return dict_tmp

# 生成语句序列
def create_sequence(dict_tmp, train_data):
    get_sent = lambda x: list(dict_tmp['id'][x])
    train_data['sent'] = list(map(get_sent, train_data['words']))
    return train_data

# 把两个xls整个在一起
def sum_xls(f1, f2):
    pass


# 训练数据处理(中文)
class ch_train_data_pretreatment:
    # 训练数据处理的对象
    def __init__(self, neg_file='data/ch_neg.xls', pos_file='data/ch_pos.xls'):
        # 训练的文本路径
        self.neg_file = neg_file
        self.pos_file = pos_file
        self.max_features = 200
        self.max_len = 50
        self.batch_size = 32
        self.sum_txt = None
        self.pos_txt = None
        self.neg_txt = None
        self.data_aggregate = None
        self.dictionary = None
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    # 创建neg_txt
    def get_neg_txt(self):
        assert os.path.exists(self.neg_file), '在data文件夹里面不存在ch_neg.xls文件'
        if not os.path.exists('tmp/ch_neg_txt.pkl'):
            clear_nan_repeat(self.neg_file)  # 去重复和去空值
            self.neg_txt = create_corpus(self.neg_file, 0)
            # 保存
            saveCache('tmp/ch_neg_txt.pkl', self.neg_txt)
        else:
            self.neg_txt = loadCache('tmp/ch_neg_txt.pkl')


    def get_pos_txt(self):
        assert os.path.exists(self.pos_file), '在data文件夹里面中不存在ch_pos.xls文件'
        if not os.path.exists('tmp/ch_pos_txt.pkl'):
            clear_nan_repeat(self.pos_file)   # 去重复和去空值
            self.pos_txt = create_corpus(self.pos_file, 1)
            # 保存
            saveCache('tmp/ch_pos_txt.pkl', self.pos_txt)
        else:
            self.pos_txt = loadCache('tmp/ch_pos_txt.pkl')

    def get_sum_txt(self):
        if self.neg_txt is None:
            self.get_neg_txt()
        if self.pos_txt is None:
            self.get_pos_txt()
        self.sum_txt = pd.concat([self.neg_txt, self.pos_txt], ignore_index=True)


    def get_dictionary(self):
        if not os.path.exists('tmp/ch_dict.pkl'):
            if self.sum_txt is None:
                self.get_sum_txt()
            self.dictionary = create_dict(self.sum_txt)
            saveCache('tmp/ch_dict.pkl', self.dictionary)
        else:
            self.dictionary = loadCache('tmp/ch_dict.pkl')

    def get_pos_sequence(self):
        if self.dictionary is not None:
            self.pos_txt = create_sequence(self.dictionary, self.pos_txt)
        else:
            self.get_dictionary()
            self.pos_txt = create_sequence(self.dictionary, self.pos_txt)

    def get_neg_sequence(self):
        if self.dictionary is not None:
            self.neg_txt = create_sequence(self.dictionary, self.neg_txt)
        else:
            self.get_dictionary()
            self.neg_txt = create_sequence(self.dictionary, self.neg_txt)

    def get_sum_sequence(self):
        if self.dictionary is not None:
            self.sum_txt = create_sequence(self.dictionary, self.sum_txt)
        else:
            self.get_dictionary()
            self.sum_txt = create_sequence(self.dictionary, self.sum_txt)

    def get_train_data(self):
        try:
            self.data_aggregate = list(sequence.pad_sequences(self.sum_txt['sent'], maxlen=self.max_len))
            self.x =  np.array(list(self.data_aggregate))
            self.y = np.array(list(self.sum_txt['mark']))
            self.x_train = self.x[::2]
            self.y_train = self.y[::2]
            self.x_test = self.x[1::2]
            self.y_test = self.y[1::2]
            self.y = np_utils.to_categorical(self.y, num_classes=2)
            self.y_train = np_utils.to_categorical(self.y_train, num_classes=2)
            self.y_test = np_utils.to_categorical(self.y_test, num_classes=2)
        except Exception as e:
            print(e)



# 训练数据处理（英文）
class en_train_data_pretreatment:


    def __init__(self, neg_file = 'data/ch_neg.xls', pos_file = 'data/pos.xls'):
        self.neg_file = neg_file
        self.pos_file = pos_file
        self.max_features = 200
        self.max_len = 50
        self.batch_size = 32
        self.sum_txt = None
        self.pos_txt = None
        self.neg_txt = None
        self.data_aggregate = None
        self.dictionary = None
        self.x_test = None
        self.y_test = None

    def get_neg_txt(self):
        assert os.path.exists(self.neg_file), '在data文件夹里面不存在en_neg.xls文件'
        if not os.path.exists('tmp/en_neg_txt.pkl'):
            clear_nan_repeat(self.neg_file) # 去重复和去空集
            self.neg_txt =

    def get_pos_txt(self):
        assert os.path.exists(self.pos_file), '在data文件夹里面不存在en_pos.xls文件'



if __name__ == '__main__':
    # dir_path = 'data/neg/'
    xls_file_name = 'data/ch_neg.xls'
    clear_nan_repeat(xls_file_name)




