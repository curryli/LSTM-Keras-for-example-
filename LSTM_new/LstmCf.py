# -*- coding: utf-8 -*-
import pandas as pd #����Pandas
import numpy as np #����Numpy
import jieba #�����ͷִ�
 
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
 

 
neg=pd.read_excel('neg.xls',header=None,index=None)
pos=pd.read_excel('pos.xls',header=None,index=None) #��ȡѵ���������
pos['mark']=1
neg['mark']=0 #��ѵ���������ϱ�ǩ
pn=pd.concat([pos,neg],ignore_index=True) #�ϲ�����
neglen=len(neg)
poslen=len(pos) #����������Ŀ
 
cw = lambda x: list(jieba.cut(x)) #����ִʺ���
pn['words'] = pn[0].apply(cw)
 
comment = pd.read_excel('sum.xls') #������������
#comment = pd.read_csv('a.csv', encoding='utf-8')
comment = comment[comment['rateContent'].notnull()] #����ȡ�ǿ�����
comment['words'] = comment['rateContent'].apply(cw) #���۷ִ� 
 
d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True) 
 
w = [] #�����д���������һ��
for i in d2v_train:
  w.extend(i)
 
dict = pd.DataFrame(pd.Series(w).value_counts()) #ͳ�ƴʵĳ��ִ���
del w,d2v_train
dict['id']=list(range(1,len(dict)+1))
 
get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent) #�ٶ�̫��
 
maxlen = 50
 
print "Pad sequences (samples x time)" 
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
 
x = np.array(list(pn['sent']))[::2] #ѵ����
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #���Լ�
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #ȫ��
ya = np.array(list(pn['mark']))
 
print 'Build model...' 
model = Sequential()
model.add(Embedding(len(dict)+1, 256))
model.add(LSTM(256, 128)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print 'Fit model...'  
model.fit(xa, ya, batch_size=32, nb_epoch=4) #ѵ��ʱ��Ϊ���ɸ�Сʱ
 
classes = model.predict_classes(xa)
acc = np_utils.accuracy(classes, ya)
print 'Test accuracy:', acc 