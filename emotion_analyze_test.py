# -*- coding=utf-8 -*-

# 测试数据集

import os
import numpy as np
import pandas as pd
import jieba
import pickle
from keras.utils import np_utils
from keras.preprocessing import sequence


from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.optimizers import Adam

# 删除Cache
def deleteCache(fileName):
    os.remove('tmp/' + fileName)

# 保存Cache
def saveCache(fileName, tmp):
    fileTmp = open('tmp/' + fileName, 'wb')
    pickle.dump(tmp, fileTmp)
    fileTmp.close()

# 加载Cache
def loadCache(fileName):
    fileTmp = open('tmp/'+ fileName, 'rb')
    tmp = pickle.load(fileTmp)
    fileTmp.close()
    return tmp

# 生成dict
def createDict(negFileName='ch_neg.xls', posFileName='pos.xls'):
    neg_txt = pd.read_excel('data/'+ negFileName, header=None, index_col=None)
    pos_txt = pd.read_excel('data/' + posFileName, header=None, index_col=None)
    pos_txt['mark'] = 1
    neg_txt['mark'] = 0
    pn = pd.concat([pos_txt, neg_txt], ignore_index=True)  # 合并语料
    negLen = len(neg_txt)
    posLen = len(pos_txt)
    print('neg:', negLen)
    print('pos:', posLen)
    cw = lambda x: list(jieba.cut(x))  # 分词函数
    pn['words'] = pn[0].apply(cw)

    d2b_train = pn['words']
    w = []
    for i in d2b_train:
        w.extend(i)
    # 统计此的出现次数
    dict = pd.DataFrame(pd.Series(w).value_counts())
    del w, d2b_train
    dict['id'] = list(range(1, len(dict) + 1))

    getSent = lambda x: list(dict['id'][x])
    pn['sent'] = pn['words'].apply(getSent)
    print('create dict Success')
    saveCache('test.pkl', tmp=pn)
    return pn



# 建立rnn神经网络
def create_simplernn_model(max_features=2000):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))

    # 学习率
    LR = 0.001
    # optimizer
    adam = Adam(LR)

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# 测试model的效果
def testModel(model, x_test, y_test, batch_size=32):
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    # print('\nTest score:', score)
    # print('\nTest accuracy:', acc)
    return acc

max_features = 2000
maxlen = 50  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

pn = 0
# deleteCache('tmp.pkl')
try:
    pn = loadCache('test.pkl')
except:
    print('No cache in tmp/. \nNow creating test data.....................................')
if pn is 0:
    try:
        pn = createDict(negFileName='ch_neg.xls', posFileName='pos.xls')
    except:
        print('load create test data error')
        exit(0)
# print(pn)

pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))


x = np.array(list(pn['sent'])) #全集
y = np.array(list(pn['mark']))
# 将0->10 1->01
y = np_utils.to_categorical(y, num_classes=2)


model = create_simplernn_model(max_features)
model.load_weights('tmp/model_weight')

acc = testModel(model, x, y)
print('\nacc: ', acc)
