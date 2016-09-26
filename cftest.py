# -*- coding: utf-8 -*-
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
 
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
 
 
neg=pd.read_excel('neg.xls',header=None,index=None)
pos=pd.read_excel('pos.xls',header=None,index=None) #读取训练语料完毕
pos['mark']=1
neg['mark']=0 #给训练语料贴上标签
pn=pd.concat([pos,neg],ignore_index=True) #合并语料
neglen=len(neg)
poslen=len(pos) #计算语料数目
 
cw = lambda x: list(jieba.cut(x)) #定义分词函数
pn['words'] = pn[0].apply(cw)
 
comment = pd.read_excel('sum.xls') #读入评论内容
#comment = pd.read_csv('a.csv', encoding='utf-8')
comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw) #评论分词 
 
d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True) 
 
w = [] #将所有词语整合在一起
for i in d2v_train:
  w.extend(i)
 
newList = list(set(w))
 
print "newlist len is"
print len(newList)
 
 
dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数


print type(dict)
print len(dict)
 