# -*- coding=utf-8 -*-

import logging
import gensim
from gensim.models import word2vec
import jieba


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

filename = 'sum'
sentences = word2vec.Text8Corpus(filename + '.fen.txt')  # 加载语料
model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5
model.save(filename+'.wc')    # 保存model

model = word2vec.Word2Vec.load('tmp/'+ filename + '.wc')    # 读取model

y1 = model.similarity('不好', '差')
print(y1)

# test_text = u'真垃圾'
# seg_list = jieba.cut(test_text)
# text_list = ' '.join(seg_list).split()
# print(text_list)
# for i in text_list

# count = []
# for i in text_list:
#     count.append(model.similarity(i, '好'))
# print(count)
# 一个字的位置
# a = model.wv['好']
# print(a)
#
# tmp = model.wv.most_similar(positive=('好'), negative=('不好') )
# print(tmp)

