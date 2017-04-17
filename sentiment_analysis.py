# -*- coding=utf-8 -*-


from handleData import ch_train_data_pretreatment
import os
import pickle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.activations import softmax, sigmoid

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

# 建立rnn神经网络
def create_simpleRNN_model(max_features=2000):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(SimpleRNN(128, dropout=0.4))
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
def trainModel(model, x_train, y_train, x_test, y_test, batch_size=32, epochs=1):
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    return model

# 测试model的效果
def testModel(model, x_test, y_test, batch_size=32):
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size,)
    return acc


# 英文训练模型
class en_train_model:
    def __init__(self):
        pass

# 中文模型
class ch_train_model:
    def __init__(self, train_neg_file='data/ch_neg.xls', train_pos_file='data/ch_pos.xls'):
        self.train_neg_file = train_neg_file
        self.train_pos_file = train_pos_file
        self.train_data = ch_train_data_pretreatment(neg_file=self.train_neg_file, pos_file=self.train_pos_file)
        self.train_data.get_sum_txt()
        self.train_data.get_dictionary()
        # 创建序列
        self.train_data.get_sum_sequence()
        # 初始化训练数据
        self.train_data.get_train_data()
        # 测试集合
        self.test_data = None
        self.model = create_simpleRNN_model()


    # 判断模型性能
    def ch_check_model(self, acc):
        if not os.path.exists('tmp/ch_result.pkl'):
            print('Saving model')
            result = dict()
            result['accuracy'] = acc
            result['model'] = self.model.to_json()
            saveCache('tmp/ch_result.pkl', result)
            self.model.save_weights('tmp/ch_model_weight')
        else:
            result = dict()
            result['accuracy'] = acc
            result['model'] = self.model.to_json()
            if loadCache('tmp/ch_result.pkl')['accuracy'] < acc:
                print('This model is better than before')
                print('Saving model')
                saveCache('tmp/ch_result.pkl', result)
                self.model.save_weights('tmp/ch_model_weight')
            else:
                print('This model is not god.')

    def train(self, epochs = 1):
        self.model = trainModel(self.model, self.train_data.x, self.train_data.y, self.train_data.x_test, self.train_data.y_test, batch_size=self.train_data.batch_size, epochs=epochs)
        acc = testModel(self.model, self.train_data.x_test, self.train_data.y_test, batch_size=self.train_data.batch_size)
        print('\n测试集的正确率：', acc)
        self.ch_check_model(acc)

    # 加载训练所得到的模型
    def load_model(self, model_file='tmp/ch_model_weight'):
        self.model.load_weights(model_file)

    def one_sentence(self, txt, mark=1):
        self.train_data.set_test_sentence(txt, mark)
        test_sentence = self.train_data.test_sentence
        result = self.model.predict(test_sentence['sent'])
        # print(result)
        return result

    def test(self):
        pass




if __name__ == '__main__':

    rnns = ch_train_model(train_neg_file='data/neg.xls', train_pos_file='data/pos.xls')

    rnns.train(20)
    rnns.load_model()
    while True:
        txt = input('请输入一句话:(no退出)')
        if txt is 'no':
            break
        else:
            result = rnns.one_sentence(txt, 1)
            print(result)
            if result[1] - result[0] >= 0.2:
                print('好评')
            else:
                print('差评')

    print('Finish')

