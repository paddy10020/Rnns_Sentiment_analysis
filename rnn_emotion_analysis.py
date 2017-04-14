# -*- coding=utf-8 -*-

import os
import numpy as np
import pandas as pd
import jieba
import pickle
from keras.utils import np_utils
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

from keras.models import Sequential,model_from_json, model_from_config, save_model, load_model
from keras.layers.core import Dense, Dropout, Activation
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
def createDict():
    neg_txt = pd.read_excel('data/neg.xls', header=None, index_col=None)
    pos_txt = pd.read_excel('data/pos.xls', header=None, index_col=None)
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
    saveCache('dict.pkl', tmp=pn)
    return pn

# 建立rnn神经网络
def createSimpleRNNModel(max_features=2000):
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

# 训练model
def trainModel(model, x_train, y_train, x_test, y_test, batch_size=20, epochs=1):
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    # Another way to train
    # BATCH_INDEX = 0
    # BATCH_SIZE = 50
    # for step in range(40001):
    #     x_batch = x[BATCH_INDEX:BATCH_INDEX + BATCH_SIZE, :]
    #     y_batch = y[BATCH_INDEX:BATCH_INDEX + BATCH_SIZE, :]
    #     cost = model.train_on_batch(x_batch, y_batch)
    #     BATCH_INDEX += BATCH_SIZE
    #     BATCH_INDEX = 0 if BATCH_INDEX >= x.shape[0] else BATCH_INDEX
    #     if step % 500 == 0:
    #         print(cost)
    return model



# 测试model的效果
def testModel(model, x_test, y_test, batch_size=32):
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    # print('\nTest score:', score)
    # print('\nTest accuracy:', acc)
    return acc


# 判断模型性能
def checkModel(model, acc, batch_size=32):
    if not os.path.exists('tmp/result.pkl'):
        print('Saving model')
        result = dict()
        result['accuracy'] = acc
        result['model'] = model.to_json()
        saveCache('result.pkl', result)
        model.save_weights('tmp/model_weight')
    else:
        result = dict()
        result['accuracy'] = acc
        result['model'] = model.to_json()
        if loadCache('result.pkl')['accuracy'] < acc:
            print('This model is better than before')
            print('Saving model')
            saveCache('result.pkl', result)
            model.save_weights('tmp/model_weight')
        else:
            print('This model is not god.')

max_features = 2000
maxlen = 50  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

pn = 0
# deleteCache('tmp.pkl')
try:
    pn = loadCache('dict.pkl')
except:
    print('No dict cache in tmp/. \nNow creating dict.....................................')
if pn is 0:
    try:
        pn = createDict()
    except:
        print('create dict error')
        exit(0)
# print(pn)

pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

# 数据序列处理
x = np.array(list(pn['sent']))[::2] #训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))
# 将0->10 1->01
y = np_utils.to_categorical(y, num_classes=2)
yt = np_utils.to_categorical(yt, num_classes=2)

model = createSimpleRNNModel(max_features)
# model = trainModel(model, x, y, xt , yt,  batch_size=batch_size, epochs=20)

# a = model.fit(x, y,
#           batch_size=batch_size,
#           epochs=1,
#           validation_data=(xt, yt))

# Another way to train
BATCH_INDEX = 0
BATCH_SIZE = 2000
acc = []
for step in range(500):
   x_batch = x[BATCH_INDEX:BATCH_INDEX + BATCH_SIZE, :]
   y_batch = y[BATCH_INDEX:BATCH_INDEX + BATCH_SIZE, :]
   cost = model.train_on_batch(x_batch, y_batch)
   BATCH_INDEX += BATCH_SIZE
   BATCH_INDEX = 0 if BATCH_INDEX >= x.shape[0] else BATCH_INDEX
   acc.append(cost[1])
   # if step % 50 == 0:
   #     print(cost[1])
ftm = open('acc.pkl', 'wb')
pickle.dump(acc, ftm)
ftm.close()
print(len(acc))

# model.load_weights('tmp/model_weight')
# acc = testModel(model, xt, yt)
# print('\nacc: ', acc)

# checkModel(model, acc)
# model.history.history.values()
# print(a.history['loss'])
# print(a.history['acc'])



# print('Finish')

