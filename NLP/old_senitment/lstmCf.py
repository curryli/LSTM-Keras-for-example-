# -*- coding: utf-8 -*-
import pandas as pd  # 导入Pandas
import numpy as np  # 导入Numpy
import jieba  # 导入结巴分词

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU


from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import recall_score, precision_score
from keras import metrics
import keras.backend as K
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import os  # python miscellaneous OS system tool

# os.chdir("C:/work/unionpay/") #changing our directory to which we have our data file
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


neg = pd.read_excel('neg.xls', header=None, index=None)
pos = pd.read_excel('pos.xls', header=None, index=None)  # 读取训练语料完毕
pos['mark'] = 1
neg['mark'] = 0  # 给训练语料贴上标签
pn = pd.concat([pos, neg], ignore_index=True)  # 合并语料
neglen = len(neg)
poslen = len(pos)  # 计算语料数目

cw = lambda x: list(jieba.cut(x))  # 定义分词函数
pn['words'] = pn[0].apply(cw)

comment = pd.read_excel('sum.xls')  # 读入评论内容
# comment = pd.read_csv('a.csv', encoding='utf-8')
comment = comment[comment['rateContent'].notnull()]  # 仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw)  # 评论分词

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index=True)

w = []  # 将所有词语整合在一起
for i in d2v_train:
    w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts())  # 统计词的出现次数
del w, d2v_train
dict['id'] = list(range(1, len(dict) + 1))

get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)  # 速度太慢

maxlen = 50

print "Pad sequences (samples x time)"
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

x = np.array(list(pn['sent']))[::2]  # 训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2]  # 测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent']))  # 全集
ya = np.array(list(pn['mark']))

print 'Build model...'
model = Sequential()
model.add(Embedding(len(dict) + 1, 256))
model.add(LSTM(128))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print 'Fit model...'
model.fit(xa, ya, batch_size=32, nb_epoch=10)  # 训练


print 'evaluate model...'
y_predict = model.predict(xt, batch_size=32)
y_predict = [j[0] for j in y_predict]
y_predict = np.where(np.array(y_predict) < 0.5, 0, 1)

precision = precision_score(yt, y_predict, average='macro')
recall = recall_score(yt, y_predict, average='macro')
print ("Precision:", precision)
print ("Recall:", recall)

confusion_matrix=confusion_matrix(yt,y_predict)
print  confusion_matrix