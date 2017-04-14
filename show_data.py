# -*- coding=utf-8 -*-
import pickle
import matplotlib.pyplot as plt

acc = []
ftm = open('acc.pkl', 'rb')
acc = pickle.load(ftm)
ftm.close()
print(len(acc))


plt.plot(acc)
plt.show()
