#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install deepctr')


# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.sequence import pad_sequences
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models.xdeepfm import xDeepFM
from sklearn.model_selection import train_test_split


# In[3]:


# importing libraries
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import xDeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names

from functools import reduce
import operator
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# compiling the model
import tensorflow as tf


# # Import Files

# In[5]:


user_log_all = pd.read_csv("user_log_all.csv")# all data import
xdeep_allbook = pd.read_csv('xdeep_allbook.csv')


# In[6]:


# customer_list
customer_list = user_log_all['customer_idx'].values.tolist()
my_set = set(customer_list) #집합set으로 변환
customer_list = list(my_set) #list로 변환


# In[7]:


customer_list


# In[8]:


xdeep_allbook.dropna(inplace=True)


# In[ ]:


def return_ctr():
    #predicting
    pred_ans_xdeep_test = model.predict(test_model_input, batch_size=256)
    #predicting
    pred_ans_xdeep_train = model.predict(train_model_input, batch_size=256)
    test['pred_ctr'] = pred_ans_xdeep_test
    train['pred_ctr'] = pred_ans_xdeep_train
    ctr = pd.concat([train,test])
    # ctr_sorted = ctr.sort_values(by=ctr.columns[-1],ascending=False)
    return ctr # final data


# # CODE

# In[24]:


cl = customer_list[:3]


# In[30]:


cl


# In[36]:


sig=0

for customer in cl:
    is_user = user_log_all['customer_idx'] == customer
    user_search = user_log_all[is_user] # user_search ; user가 검색한 param 값들

    param_list = user_search['param'].values.tolist()
    # param_list # user의 keyword검색값, isbn검색값
    
    # # 사용자 검색어(param)를 키워드 or isbn으로 분리 # #
    word = ("0", "1","2","3","4","5","6","7","8","9")
    param_only = []
    param_isbn = []

    def param_filtering(x):
        if x.startswith(word) and x.endswith(word):
            param_isbn.append(x)
        else:
            param_only.append(x)

    # 사용자 검색어(param)를 키워드 or isbn으로 분리
    user_search['param'].apply(param_filtering)
    
    
    # # param_isbn검색 있을시, param_isbn을 띄어쓰기 기준으로 리스트로 만듬. # #
    isbn_after_filter_1 = []
    isbn_int = []

    if param_isbn != []:
        for x in param_isbn:
            x_split = x.split(' ')
            isbn_after_filter_1.append(x_split)

        isbn_after_filter_2 = list(reduce(operator.add, isbn_after_filter_1))

        isbn_after_filter_2 = [word.strip('X') for word in isbn_after_filter_2 ] # X 제거
        isbn_after_filter_2 = [word.strip('-') for word in isbn_after_filter_2 ] # - 제거
        isbn_after_filter_2 = [word.strip(',') for word in isbn_after_filter_2 ] # , 제거
        isbn_after_filter_2 = [word.strip('.') for word in isbn_after_filter_2 ] # . 제거
        isbn_after_filter_2 = list(filter(None, isbn_after_filter_2)) # ''제거

        # - 제거
        isbn_after_filter_3 = []
        for a in isbn_after_filter_2:
            isbn_after_filter_3.append(a.replace('-',''))

        isbn_int = list(map(int,isbn_after_filter_3)) # 정수로 변환
        
        # # 사용자 검색책들인 searchdata 찾기 # #
        
        searchdata=xdeep_allbook[:1]

    for i,param in enumerate(param_only):
        param_mask = xdeep_allbook['title'].str.contains(param)
        search_param = xdeep_allbook[param_mask] # search_param ; 사용자가 검색한 kwd 책 리스트
        isbn_mask = xdeep_allbook['isbn'].isin(isbn_int)
        search_isbn = xdeep_allbook[isbn_mask] # search_isbn ; 사용자가 검색한 isbn 책 리스트

        if i==1: # i==1일때는 첫번째 데이터프레임을 갖고옴
            tmp = pd.concat([search_param,search_isbn])
            searchdata = tmp
        else: # i==2부터는 첫번째 데이터프레임에 concat
            tmp = pd.concat([search_param,search_isbn])
            searchdata = pd.concat([searchdata,tmp])
            
    # searchdata 전처리
    searchdata.drop(['title.1'], axis=1,inplace=True)
    searchdata['Cn_3'].dropna(inplace=True) # Cn_3 결측값 제거
    searchdata.drop_duplicates(['isbn'],inplace=True) # 중복제거
    user_log_cn3 = searchdata['Cn_3'].values.tolist() # user_log_cn3 ; user가 검색한 책들의 Cn3_list
    
    
    
    # # user가 가장 관심있는 category best 3 추출 # #
    counter = Counter(user_log_cn3)
    keycount = dict(counter) # keycount ; 빈도수 dict
    sorted_cn3 = sorted(keycount.items(), reverse=True,key=lambda item: item[1])
    cn3_user_interestd = dict(sorted_cn3[:3]) # 유저가 가장 관심있는 카테고리
    
    
    
    # # user log 를 바탕으로 negative / positive data 생성 # #
    #-user가 검색한 책 카테고리에 속한 것은 1로 labeling
    #-user가 검색하지 않은 책 카테고리에 속한 것은 0으로 labeling
    userlog_cn3_mask = xdeep_allbook['Cn_3'].isin(user_log_cn3)
    negative_data = xdeep_allbook[~userlog_cn3_mask]
    positive_data = xdeep_allbook[userlog_cn3_mask]

    # data labeling
    positive_data['label']=1
    negative_data['label']=0

    # data ; 해당 customer의 labeling data
    data = pd.concat([positive_data,negative_data])
    data['customer_id'] = customer
    data.drop_duplicates(['isbn'],inplace=True) # 중복제거
    #data # final data
    print("customer:",customer)
    
    if(sig==0):
        data_final = data
        sig=sig+1
    else:
        df = pd.concat([data_final,data],ignore_index=True)


# In[40]:


df


# In[55]:


data = df


# In[59]:


# categorising the features into sparse/dense feaure set
sparse_features = ["customer_id","isbn","author", "publisher","date"]
dense_features = ['catev_' + str(i) for i in range(0,80)]
dense_features2 = [str(i) for i in range(0,256)]
dense_features.extend(dense_features2)


# data inputation for missing values
data[sparse_features] = data[sparse_features].fillna('-1',)

# creating target variable
target = ['label']

# encoding function  // labelencoder ; 문자를 0부터 증가하는 정수형 숫자로 바꿔주는기능을 제공
def encoding(data,feat,encoder):
    data[feat] = encoder.fit_transform(data[feat])

# encoding for categorical feautures
[encoding(data, feat, LabelEncoder()) for feat in ["author","publisher"]]


# creating a 4 bit embedding for every sparse feature
sparse_feature_columns = [SparseFeat(feat,vocabulary_size = data[feat].nunique(), embedding_dim=4)
                         for i,feat in enumerate(sparse_features)]


# Creating a dense feat
dense_feature_columns = [DenseFeat(feat,1) for feat in dense_features]



# features to be used for dnn part of xdeepfm
dnn_feature_columns = sparse_feature_columns + dense_feature_columns
# features to be used for linear part of xdeepfm
linear_feature_columns = sparse_feature_columns + dense_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
feature_names


# In[61]:


# creating train test splits
train, test = train_test_split(data, test_size = 0.2) # 200*0.2 = 40개를 test_dataset으로 분리
train_model_input = {name: train[name].values for name in feature_names}
test_model_input = {name: test[name].values for name in feature_names}



model = xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256),
cin_layer_size=(128, 128),
cin_split_half=True, cin_activation='relu'
,l2_reg_linear=1e-05,
l2_reg_embedding=1e-05, l2_reg_dnn=0, l2_reg_cin=0,
seed=1024, dnn_dropout=0,dnn_activation='relu',
dnn_use_bn=False, task='binary')

model.compile(optimizer='adam',loss= 'mse', 
              metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])


# In[63]:


# training the model
history = model.fit(train_model_input, train['label'].values,
                    batch_size = 128, epochs = 2, verbose = 2, validation_split = 0.2, )


# In[64]:


ctr = return_ctr()


# In[65]:


ctr


# # ctr바탕으로 추천 구현

# ## (유저가 검색한 책 카테고리3 으로 필터링)

# 1. useridx해당 dataframe갖고오기
# 2. user가 가장 관심있는 bestcategory찾기 -> 카테고리 필터링
# 3. 필터링된 df 바탕으로 ctr sort후 예측 

# In[72]:


def find_user_interest_category(customer_idx):
    is_user = user_log_all['customer_idx'] == customer_idx
    user_search = user_log_all[is_user] # user_search ; user가 검색한 param 값들

    param_list = user_search['param'].values.tolist()
    param_list # user의 keyword검색값, isbn검색값

    ##### 사용자 검색어를 키워드 or isbn으로 분리 #####
    word = ("0", "1","2","3","4","5","6","7","8","9")
    param_only = []
    param_isbn = []

    def param_filtering(x):
        if x.startswith(word) and x.endswith(word):
            param_isbn.append(x)
        else:
            param_only.append(x)

    # 사용자 검색어(param)를 키워드 or isbn으로 분리
    user_search['param'].apply(param_filtering)

     # param_isbn검색 있을시, param_isbn을 띄어쓰기 기준으로 리스트로 만듬.
    isbn_after_filter_1 = []
    isbn_int = []

    if param_isbn != []:
        for x in param_isbn:
            x_split = x.split(' ')
            isbn_after_filter_1.append(x_split)

        isbn_after_filter_2 = list(reduce(operator.add, isbn_after_filter_1))

        isbn_after_filter_2 = [word.strip('X') for word in isbn_after_filter_2 ] # X 제거
        isbn_after_filter_2 = [word.strip('-') for word in isbn_after_filter_2 ] # - 제거
        isbn_after_filter_2 = [word.strip(',') for word in isbn_after_filter_2 ] # , 제거
        isbn_after_filter_2 = [word.strip('.') for word in isbn_after_filter_2 ] # . 제거
        isbn_after_filter_2 = list(filter(None, isbn_after_filter_2)) # ''제거

        # - 제거
        isbn_after_filter_3 = []
        for a in isbn_after_filter_2:
            isbn_after_filter_3.append(a.replace('-',''))

        isbn_int = list(map(int,isbn_after_filter_3)) # 정수로 변환


    ##### 사용자 검색책들인 searchdata 찾기 #####
    searchdata=xdeep_allbook[:1]

    for i,param in enumerate(param_only):
        param_mask = xdeep_allbook['title'].str.contains(param)
        search_param = xdeep_allbook[param_mask] # search_param ; 사용자가 검색한 kwd 책 리스트
        isbn_mask = xdeep_allbook['isbn'].isin(isbn_int)
        search_isbn = xdeep_allbook[isbn_mask] # search_isbn ; 사용자가 검색한 isbn 책 리스트

        if i==1: # i==1일때는 첫번째 데이터프레임을 갖고옴
            tmp = pd.concat([search_param,search_isbn])
            searchdata = tmp
        else: # i==2부터는 첫번째 데이터프레임에 concat
            tmp = pd.concat([search_param,search_isbn])
            searchdata = pd.concat([searchdata,tmp])

    # searchdata 전처리
    searchdata.drop(['title.1'], axis=1,inplace=True)
    searchdata['Cn_3'].dropna(inplace=True) # Cn_3 결측값 제거
    searchdata.drop_duplicates(['isbn'],inplace=True) # 중복제거
    user_log_cn3 = searchdata['Cn_3'].values.tolist() # user_log_cn3 ; user가 검색한 책들의 Cn3_list

    ##### user가 가장 관심있는 category best 3 추출 #####
    counter = Counter(user_log_cn3)
    keycount = dict(counter) # keycount ; 빈도수 dict
    sorted_cn3 = sorted(keycount.items(), reverse=True,key=lambda item: item[1])
    cn3_user_interested = dict(sorted_cn3[:3]) # 유저가 가장 관심있는 카테고리
    
    return cn3_user_interested


# In[85]:


def book_recommend_func(customer_idx):
    is_user = ctr['customer_id'] == customer_idx
    user_ctr = ctr[is_user] # user_ctr ; 해당 user에 대한 전체 책 CTR예측표 가져오기
     
    # user관심 카테고리
    cn3_user_interested =  find_user_interest_category(customer_idx)
    
    # Cn3 filtering
    userlike = user_ctr['Cn_3'].isin(cn3_user_interestd) #해당 유저가 관심있는 카테고리기반 필터링
    user_ctr_like = user_ctr[userlike]
    ctr_sorted = user_ctr_like.sort_values(by=['pred_ctr'],ascending=False)
    
    res = ctr_sorted[['isbn','customer_id','title','pred_ctr']]

    return res[:30]


# In[86]:


book_recommend_func(46)


# In[80]:


find_user_interest_category(46) # 46의 like cate


# In[ ]:




