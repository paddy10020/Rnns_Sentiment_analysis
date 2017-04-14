# -*- coding=utf-8 -*-

import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
import pickle
from keras.utils import np_utils


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN


# neg=pd.read_excel('data/neg.xls',header=None,index=None)
# pos=pd.read_excel('data/pos.xls',header=None,index=None) #读取训练语料完毕
# pos['mark']=1
# neg['mark']=0 #给训练语料贴上标签
# pn=pd.concat([pos,neg],ignore_index=True) #合并语料
# neglen=len(neg)
# poslen=len(pos) #计算语料数目
#
# cw = lambda x: list(jieba.cut(x)) #定义分词函数
# pn['words'] = pn[0].apply(cw)
#
# comment = pd.read_excel('data/sum.xls') #读入评论内容
# comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
# comment['words'] = comment['rateContent'].apply(cw) #评论分词
#
# d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True)
#
# w = [] #将所有词语整合在一起
# for i in d2v_train:
#   w.extend(i)
#
# dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
# del w,d2v_train
# dict['id']=list(range(1,len(dict)+1))
#
# get_sent = lambda x: list(dict['id'][x])
# pn['sent'] = pn['words'].apply(get_sent) #速度太慢
#
max_features = 2000
maxlen = 50  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
#
# print("Pad sequences (samples x time)")
# pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

# f1 = open('dict.plk', 'wb')
# pickle.dump(pn, f1)
# pickle.dump(dict, f1)
# f1.close()

f2 = open('dict.plk', 'rb')
pn = pickle.load(f2)
dict = pickle.load(f2)
f2.close()


x = np.array(list(pn['sent']))[::2] #训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))
y = np_utils.to_categorical(y, num_classes=2)
yt = np_utils.to_categorical(yt, num_classes=2)


print(x.shape)
print(xt.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x, y,
          batch_size=batch_size,
          epochs=5,
          validation_data=(xt, yt))

score, acc = model.evaluate(xt, yt, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)


