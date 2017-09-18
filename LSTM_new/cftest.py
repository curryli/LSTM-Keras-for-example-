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
 
newList = list(set(w))
 
print "newlist len is"
print len(newList)
 
 
dict = pd.DataFrame(pd.Series(w).value_counts()) #ͳ�ƴʵĳ��ִ���


print type(dict)
print len(dict)
 