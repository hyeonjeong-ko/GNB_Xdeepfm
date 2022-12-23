#!/usr/bin/env python
# coding: utf-8

# # module & data import

# In[1]:


get_ipython().system('pip install deepctr')


# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.sequence import pad_sequences
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models.xdeepfm import xDeepFM
from sklearn.model_selection import train_test_split


# In[2]:


# importing libraries
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import xDeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names

from functools import reduce
import operator
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# compiling the model
import tensorflow as tf


# In[5]:


def choose_univ(univ_name):
    if(univ_name == "University of Michigan"):
        res = univlog_1
    elif(univ_name == "University of California Berkeley"):
        res = univlog_2
    elif(univ_name == "George Washington University"):
        res = univlog_3
    return res


# In[6]:


# 최종 input data 생성(df)
def data_pca(data):
   # 차원축소할 data
   dense_features = ['catev_' + str(i) for i in range(1,80)]
   dense_features2 = [str(i) for i in range(1,80)]
   dense_features.extend(dense_features2)
   
   # PCA 적용 - 컴포넌트는 1로 설정합니다.
   df_scaled = StandardScaler().fit_transform(data[dense_features])


   pca = PCA(n_components=5)
   #fit( )과 transform( ) 을 호출하여 PCA 변환 데이터 반환
   pca.fit(df_scaled)
   df_pca = pca.transform(df_scaled)
   
   df = pd.DataFrame(df_pca,columns=['pca_vector' + str(i) for i in range(1,6)]) # 차원축소 dataframe생성
   
   tmpcol =  ["title","isbn","author", "publisher","date",'label','Cn_1','Cn_2','Cn_3','Cn_4']
   df[tmpcol] = data[tmpcol]
   return df


# In[7]:


def Xdeep_encoding(data):
    # data inputation for missing values
    data[sparse_features] = data[sparse_features].fillna('-1',)

    # encoding function  // labelencoder ; 문자를 0부터 증가하는 정수형 숫자로 바꿔주는기능을 제공
    def encoding(data,feat,encoder):
        data[feat] = encoder.fit_transform(data[feat])

    # encoding for categorical feautures
    [encoding(data, feat, LabelEncoder()) for feat in sparse_features]

    # 1.Label Encoding for sparse features
    for feat in sparse_features:
      lbe = LabelEncoder()
      data[feat] = lbe.fit_transform(data[feat])
    
    return data


# In[8]:


def return_ctr():
    #predicting
    pred_ans_xdeep_test = model.predict(test_model_input, batch_size=256)
    #predicting
    pred_ans_xdeep_train = model.predict(train_model_input, batch_size=256)
    test['pred_ctr'] = pred_ans_xdeep_test
    train['pred_ctr'] = pred_ans_xdeep_train
    ctr = pd.concat([train,test])
    ctr_sorted = ctr.sort_values(by=ctr.columns[-1],ascending=False)
    return ctr_sorted # final data


# In[9]:


def Cn3_filtering():
    # gnb에서 검색 로그찾기
    gnb_univ = gnb[gnb['company_name'].str.contains(univname)]
    user_log_cn3 = gnb_univ['Cn_3'].values.tolist() # user_log_cn3 ; user가 검색한 책들의 Cn3_list
    counter = Counter(user_log_cn3)
    keycount = dict(counter)
    sorted_cn3 = sorted(keycount.items(), reverse=True,key=lambda item: item[1])
    
    cn3_user_interestd = dict(sorted_cn3[:3])
    
    userlike = ctr_sorted['Cn_3'].isin(cn3_user_interestd)
    like = ctr_sorted[userlike]
    return like # 유저가 가장 관심있는 카테고리


# In[ ]:





# In[4]:


# all data import
univlog_1 = pd.read_csv('univlog_1.csv')
univlog_2 = pd.read_csv('univlog_2.csv')
univlog_3 = pd.read_csv('univlog_3.csv')
gnb = pd.read_csv('gnbgnb.csv') # user검색 Cn3 빈도수 얻기 위해 import


# In[10]:


univ1_recommend = pd.read_csv('univ1_recommend.csv')
univ2_recommend = pd.read_csv('univ2_recommend.csv')
univ3_recommend = pd.read_csv('univ3_recommend.csv')


# # University1에 대한 추천 결과

# In[15]:


cn3_univ1_list = univlog_1['Cn_3'].values.tolist()


# In[26]:


def Cn3_filtering(univname):
    # gnb에서 검색 로그찾기
    gnb_univ = gnb[gnb['company_name'].str.contains(univname)]
    user_log_cn3 = gnb_univ['Cn_3'].values.tolist() # user_log_cn3 ; user가 검색한 책들의 Cn3_list
    counter = Counter(user_log_cn3)
    keycount = dict(counter)
    sorted_cn3 = sorted(keycount.items(), reverse=True,key=lambda item: item[1])
    
    return sorted_cn3


# In[27]:


u1  = Cn3_filtering('University of Michigan')


# In[47]:


gnb[gnb['company_name']=='George Washington University']


# In[43]:


univlog_1[univlog_1['label']==1]


# In[28]:


u1


# In[11]:


univ1_recommend


# # University2에 대한 추천 결과

# In[32]:


u2  = Cn3_filtering('University of California Berkeley')


# In[33]:


u2


# In[34]:


univ2_recommend


# # University3에 대한 추천 결과

# In[37]:


u3  = Cn3_filtering('George Washington University')


# In[39]:


u3


# In[40]:


univ3_recommend


# In[ ]:




