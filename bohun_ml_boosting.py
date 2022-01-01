#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycaret.classification import *
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import random
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


bohun= pd.read_csv('ML_430.csv', encoding='cp949')
display(bohun.head())


# In[3]:


nrow = bohun.shape[0] # 환자 명수 데이터 양(row)
ncol = bohun.shape[1] # 피쳐 갯수(column)
print('nrow: %d' % nrow, 'ncol: %d' % ncol )


# In[4]:


feature_columns_a= list(bohun.columns[4:76]) ##pet1 feature   
len(feature_columns_a) 


# In[5]:


feature_columns_b= list(bohun.columns[76:149]) ## pet2 feature
len(feature_columns_b)


# In[6]:


x_columns_1 = feature_columns_a    ## 72개 주요 pet1 feature 반영 
x_columns_2 = feature_columns_a + feature_columns_b ## 144개 변수 pet1+et2 
x_columns_3=feature_columns_b
bohun_train_1=bohun[x_columns_1] ## dataframe 형태 pet1 변수만 반영 <72개>
bohun_train_2=bohun[x_columns_2] ## dataframe 형태 pet1,pet2 변수 모두 반영 <144개> 
bohun_train_3=bohun[x_columns_3] ## dataframe 형태 pet2 변수만 반영 <72개> 


# In[7]:


bohun_y_train_1=bohun[bohun.columns[3]] # pathologic CR a


# ## 변수 스케일링

# In[8]:


from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()


# In[9]:


## bohun_train_1 (72개 pet1 기본 변수만 적용한 트레인 셋 스케일 조정)
bohun_train_1[bohun.columns[4:76]] = standardScaler.fit_transform(bohun_train_1[bohun.columns[4:76]])
bohun_train_2[bohun.columns[4:76]]=standardScaler.fit_transform(bohun_train_2[bohun.columns[4:76]])
bohun_train_2[bohun.columns[76:149]]=standardScaler.fit_transform(bohun_train_2[bohun.columns[76:149]])
bohun_train_3[bohun.columns[76:149]]=standardScaler.fit_transform(bohun_train_3[bohun.columns[76:149]])


# In[10]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(bohun_train_1,bohun_y_train_1, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[11]:


import seaborn as sns


# In[12]:


sns.countplot(x="pcr", data=bohun)
plt.title("분포확인")
plt.show()


# In[13]:


df_train_y=pd.DataFrame(train_y_1)
df_test_y=pd.DataFrame(test_y_1)


# In[14]:


sns.countplot(x="pcr", data=df_train_y)
plt.title("test count")
plt.show()


# In[15]:


sns.countplot(x="pcr", data=df_test_y)
plt.title("test count")
plt.show()


# In[16]:


import seaborn as sns


# In[17]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[18]:


random_forest_model1 = RandomForestClassifier(n_estimators = 600, # 900번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 40) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행
predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[19]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# ### SMOTE적용

# In[20]:


from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot


# In[21]:


# 모델설정
sm=SMOTE(random_state=0)
# train데이터를 넣어 복제함
X_resampled_1, y_resampled_1 = sm.fit_sample(bohun_train_1,bohun_y_train_1)


# In[22]:


def count_and_plot(y): 
    counter = Counter(y)
    for k,v in counter.items():
        print('Class=%d, n=%d (%.3f%%)' % (k, v, v / len(y) * 100))
    pyplot.bar(counter.keys(), counter.values())
    pyplot.show()
    


# In[23]:


count_and_plot(y_resampled_1)


# In[24]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(X_resampled_1,y_resampled_1, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[25]:


random_forest_model1 = RandomForestClassifier(n_estimators = 600, # 900번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 40) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행
predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[26]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# #### -pet1 cv모델링

# In[27]:


rf= RandomForestClassifier(random_state=45)


# In[28]:


from scipy import stats


# In[29]:


##random search (구간 정해서)
params={'n_estimators' : stats.randint(200,500),
       'max_features' : stats.randint(4,8),
        'min_samples_leaf' :stats.randint(1,5)}


# In[30]:


from sklearn.model_selection import RandomizedSearchCV


# In[31]:


rand_cv=RandomizedSearchCV(rf,param_distributions=params,n_iter=10,cv=10,random_state=45)


# In[32]:


rand_cv.fit(train_1,train_y_1)


# In[33]:


print('최적의 하이퍼파라미터: ', rand_cv.best_params_) 


# In[34]:


print('score: ', rand_cv.best_score_)


# In[35]:


predict1 = rand_cv.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[48]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,rand_cv.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[77]:


random_forest_model1 = RandomForestClassifier(n_estimators =381, # 900번 추정
                                             max_depth = 6,
                                              min_samples_leaf=1,
                                    
                                              # 트리 최대 깊이 5
                                             random_state = 40) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행
predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[80]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[38]:


## pet1-RF


# In[79]:


model1.predict_proba(test_1)[:, 0]


# In[66]:


list(test_y_1)


# In[ ]:





# In[ ]:





# ### pet 1- ROC커브 그리기 

# In[176]:


from sklearn.metrics import plot_roc_curve

def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    plt.figure(figsize=(10,8))
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)
    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set_facecolor((0.96, 0.96, 0.96))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate (1-Sensitivity )')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend()
    plt.show()


# In[177]:


random_forest_model1 = RandomForestClassifier(n_estimators = 381, # 900번 추정
                                             max_features = 6, # 트리 최대 깊이 5
                                            min_samples_leaf= 1,
                                             random_state = 45) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행


# In[178]:


roc_curve_plot(test_y_1,model1.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# #### LGBM model

# In[46]:


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}


# In[47]:


clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, 
    param_distributions=param_test, 
    n_iter=10,
    scoring='roc_auc',
    cv=10,
    refit=True,
    random_state=314,
    verbose=True)


# In[48]:


model1=gs.fit(train_1, train_y_1)
predict1= model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[49]:


print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


# In[50]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['LGBM: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['LGBM: Precision'] = precision_score(test_y_1, predict1)
model_metrics['LGBM: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['LGBM: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics[': AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[51]:


len(model1.predict_proba(test_1)[:, 1])


# In[55]:


model1.predict_proba(test_1)[:, 0]


# In[54]:


list(test_y_1)


# In[69]:


len(model1.predict(test_1))


# In[284]:


roc_curve_plot(test_y_1,model1.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# ### XGBM 모델링

# In[67]:


from xgboost import XGBClassifier


# In[68]:


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 random_state=45)


# In[69]:


param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}


# In[70]:


gsearch1= GridSearchCV(estimator = xgb1,param_grid = param_test1, scoring='roc_auc',iid=False, cv=10)


# In[71]:


model1=gsearch1.fit(train_1, train_y_1)
predict1= model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[72]:


print('Best score reached: {} with params: {} '.format(gsearch1.best_score_, gsearch1.best_params_))


# In[74]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['XGB: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['XGB: Precision'] = precision_score(test_y_1, predict1)
model_metrics['XGB: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['XGB: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['XGB: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[75]:


model1.predict_proba(test_1)[:, 1]


# In[76]:


list(test_y_1)


# In[303]:


roc_curve_plot(test_y_1,model1.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# ## pet 2

# In[86]:


# 모델설정
sm=SMOTE(random_state=0)
# train데이터를 넣어 복제함
X_resampled_3, y_resampled_3 = sm.fit_sample(bohun_train_3,bohun_y_train_1)


# In[87]:


count_and_plot(y_resampled_3)


# In[88]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(X_resampled_3,y_resampled_3, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[89]:


random_forest_model1 = RandomForestClassifier(n_estimators = 600, # 900번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 40) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행
predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[90]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# ### -pet2 CV모델링

# In[91]:


import scipy.stats as stats 
from sklearn.model_selection import RandomizedSearchCV


# In[92]:


rf= RandomForestClassifier(random_state=45)


# In[93]:


##random search (구간 정해서)
params={'n_estimators' : stats.randint(200,500),
       'max_features' : stats.randint(4,8),
        'min_samples_leaf' :stats.randint(1,5)}


# In[94]:


rand_cv=RandomizedSearchCV(rf,param_distributions=params,n_iter=10,cv=10,random_state=45)


# In[95]:


rand_cv.fit(train_1,train_y_1)


# In[96]:


print('score: ', rand_cv.best_score_)


# In[97]:


print('최적의 하이퍼파라미터: ', rand_cv.best_params_) 


# In[98]:


predict1 = rand_cv.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[99]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,rand_cv.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[100]:


rand_cv.predict_proba(test_1)[:, 1]


# In[101]:


list(test_y_1)


# In[ ]:





# ### pet 2- ROC커브 그리기 

# In[102]:


from sklearn.metrics import plot_roc_curve
def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    plt.figure(figsize=(10,8))
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)
    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set_facecolor((0.96, 0.96, 0.96))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate (1-Sensitivity )')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend()
    plt.show()


# In[103]:


roc_curve_plot(test_y_1,rand_cv.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# ### LGBM모델링

# In[104]:


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}


# In[105]:


clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, 
    param_distributions=param_test, 
    n_iter=10,
    scoring='roc_auc',
    cv=10,
    refit=True,
    random_state=314,
    verbose=True)


# In[106]:


model1=gs.fit(train_1, train_y_1)
predict1= model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[107]:


print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


# In[108]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['LGBM: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['LGBM: Precision'] = precision_score(test_y_1, predict1)
model_metrics['LGBM: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['LGBM: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics[': AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[109]:


roc_curve_plot(test_y_1,model1.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# In[110]:


model1.predict_proba(test_1)[:, 1]


# In[ ]:





# ### XGB 모델링

# In[111]:


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 random_state=45)


# In[112]:


param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}


# In[113]:


gsearch1= GridSearchCV(estimator = xgb1,param_grid = param_test1, scoring='roc_auc',iid=False, cv=10)


# In[114]:


model1=gsearch1.fit(train_1, train_y_1)
predict1= model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[115]:


print('Best score reached: {} with params: {} '.format(gsearch1.best_score_, gsearch1.best_params_))


# In[116]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['XGB: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['XGB: Precision'] = precision_score(test_y_1, predict1)
model_metrics['XGB: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['XGB: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['XGB: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[117]:


roc_curve_plot(test_y_1,model1.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# In[118]:


model1.predict_proba(test_1)[:, 1]


# In[ ]:





# In[ ]:





# ## PET1,2

# In[119]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(bohun_train_2,bohun_y_train_1, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[120]:


random_forest_model1 = RandomForestClassifier(n_estimators = 600, # 900번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 40) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행
predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[121]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# ## SMOTE 적용

# In[122]:


# 모델설정
sm=SMOTE(random_state=0)
# train데이터를 넣어 복제함
X_resampled_2, y_resampled_2 = sm.fit_sample(bohun_train_2,bohun_y_train_1)


# In[123]:


count_and_plot(y_resampled_2)


# In[124]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(X_resampled_2,y_resampled_2, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[125]:


random_forest_model1 = RandomForestClassifier(n_estimators = 600, # 900번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 40) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행
predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[126]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# #### -pet1,2 10cv modeling

# In[127]:


rf= RandomForestClassifier(random_state=45)


# In[128]:


##random search (구간 정해서)
params={'n_estimators' : stats.randint(200,500),
       'max_features' : stats.randint(4,8),
        'min_samples_leaf' :stats.randint(1,5)}


# In[129]:


rand_cv=RandomizedSearchCV(rf,param_distributions=params,n_iter=10,cv=10,random_state=45)


# In[130]:


rand_cv.fit(train_1,train_y_1)


# In[131]:


predict1 = rand_cv.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[132]:


print('최적의 하이퍼파라미터: ', rand_cv.best_params_) 


# In[133]:


print('score: ', rand_cv.best_score_)


# In[134]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,rand_cv.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[137]:


rand_cv.predict_proba(test_1)[:, 1]


# In[138]:


list(test_y_1)


# ### pet 1,2- ROC커브 그리기 

# In[135]:


from sklearn.metrics import plot_roc_curve
def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    plt.figure(figsize=(10,8))
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)
    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set_facecolor((0.96, 0.96, 0.96))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate (1-Sensitivity )')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend()
    plt.show()


# In[136]:


roc_curve_plot(test_y_1,rand_cv.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# ### LGBM 모델링

# In[139]:


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}


# In[140]:


clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, 
    param_distributions=param_test, 
    n_iter=10,
    scoring='roc_auc',
    cv=10,
    refit=True,
    random_state=314,
    verbose=True)


# In[141]:


model1=gs.fit(train_1, train_y_1)
predict1= model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[142]:


print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


# In[143]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['LGBM: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['LGBM: Precision'] = precision_score(test_y_1, predict1)
model_metrics['LGBM: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['LGBM: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics[': AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[144]:


model1.predict_proba(test_1)[:, 1]


# In[ ]:





# In[145]:


roc_curve_plot(test_y_1,model1.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# In[146]:


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 random_state=45)


# In[147]:


param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}


# In[148]:


gsearch1= GridSearchCV(estimator = xgb1,param_grid = param_test1, scoring='roc_auc',iid=False, cv=10)


# In[149]:


model1=gsearch1.fit(train_1, train_y_1)
predict1= model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[150]:


print('Best score reached: {} with params: {} '.format(gsearch1.best_score_, gsearch1.best_params_))


# In[151]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['XGB: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['XGB: Precision'] = precision_score(test_y_1, predict1)
model_metrics['XGB: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['XGB: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['XGB: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# In[152]:


model1.predict_proba(test_1)[:, 1]


# In[153]:


roc_curve_plot(test_y_1,model1.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# ### xgb modeling- feature importance

# In[350]:


model1=XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 random_state=45)
model1=model1.fit(train_1, train_y_1)


# In[351]:


from xgboost import plot_importance
from matplotlib import pyplot
plt.rcParams["figure.figsize"] = (5,25)
plot_importance(model1)
pyplot.show()


# In[352]:


from xgboost import plot_importance
from matplotlib import pyplot
plt.rcParams["figure.figsize"] = (5,25)
plot_importance(model1,max_num_features=30)
pyplot.show()


# ### threshold 구간에 따른 변수 갯수에 의한 성능 확인

# In[348]:


# Fit model using each importance as a threshold
from sklearn.feature_selection import SelectFromModel
from numpy import sort
thresholds = sort(model1.feature_importances_)


# In[349]:


for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model1, threshold=thresh, prefit=True)
    select_X_train = selection.transform(train_1)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, train_y_1)
    # eval model
    select_X_test = selection.transform(test_1)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_y_1, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


# ## 상위 feature importance 수치 정리

# In[132]:


##random serach 모델 고정 *pet1,2, 10cv기준;
random_forest_model1 = RandomForestClassifier(n_estimators = 381, # 900번 추정
                                             max_features = 6, # 트리 최대 깊이 5
                                             min_samples_leaf=1,
                                             random_state = 45) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행


# In[133]:


model1.feature_importances_  ## 모델링 feature importance 값들 확인 


# In[134]:


list_column = []   ##컬럼 리스트
list_fi = []   ##featureimportance 리스트
for i,j in zip(X_resampled_2.columns,model1.feature_importances_):
    list_column.append(i)
    list_fi.append(j)


# In[135]:


## feature importance 시각화 
plt.rcParams["figure.figsize"] = (5,25)
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(list_fi)), list_fi, color='b', align='center')
plt.yticks(range(len(list_column)), list_column)
plt.xlabel('Relative Importance')


# In[136]:


## feature importance 상위 피쳐들 선택해서 df(데이터 프레임)으로 만드는 작업
df_importance = pd.DataFrame(list_column, columns=['list_column'])
df_importance


# In[137]:


df_importance['list_fi'] = list_fi


# In[138]:


df_importance.sort_values('list_fi',ascending=False)   ##list_fi 값대로 descending (sort_values)


# In[139]:


##feature importance 순으로 상위 30개만 추출해서 df 반환 
df_importance.sort_values('list_fi',ascending=False)[:30]


# In[140]:


##상위 30개 변수들 리스트
columnlist_top30=df_importance.sort_values('list_fi',ascending=False)[:30].list_column.tolist()
columnlist_top20=df_importance.sort_values('list_fi',ascending=False)[:20].list_column.tolist()
columnlist_top10=df_importance.sort_values('list_fi',ascending=False)[:10].list_column.tolist()


# In[141]:


df_top30 = X_resampled_2[columnlist_top30] #Top30 columnlist 해당되는 값들 리스트만 df_top30에 저장, split이전
df_top20 = X_resampled_2[columnlist_top20]
df_top10 = X_resampled_2[columnlist_top10]


# In[142]:


columnlist_top30


# ## Top30 모델링

# In[189]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(df_top30,y_resampled_2, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[145]:


random_forest_model1 = RandomForestClassifier(n_estimators = 381, # 900번 추정
                                             max_features = 6, # 트리 최대 깊이 5
                                             min_samples_leaf=1,
                                             random_state = 45) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행


# In[146]:


predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[147]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# ### Top30- ROC커브 그리기 

# In[190]:


from sklearn.metrics import plot_roc_curve
def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    plt.figure(figsize=(10,8))
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)
    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set_facecolor((0.96, 0.96, 0.96))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate (1-Sensitivity )')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend()
    plt.show()


# In[191]:


random_forest_model1 = RandomForestClassifier(n_estimators = 381, # 900번 추정
                                             max_features = 6, # 트리 최대 깊이 5
                                            min_samples_leaf= 1,
                                             random_state = 45) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행


# In[192]:


roc_curve_plot(test_y_1,model1.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# In[193]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(df_top20,y_resampled_2, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[149]:


random_forest_model1 = RandomForestClassifier(n_estimators = 381, # 900번 추정
                                             max_features = 6, # 트리 최대 깊이 5
                                             min_samples_leaf=1,
                                             random_state = 45) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행


# In[150]:


predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[151]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# ### Top20- ROC커브 그리기 

# In[194]:


from sklearn.metrics import plot_roc_curve
def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    plt.figure(figsize=(10,8))
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)
    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set_facecolor((0.96, 0.96, 0.96))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate (1-Sensitivity )')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend()
    plt.show()


# In[195]:


random_forest_model1 = RandomForestClassifier(n_estimators = 381, # 900번 추정
                                             max_features = 6, # 트리 최대 깊이 5
                                            min_samples_leaf= 1,
                                             random_state = 45) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행


# In[196]:


roc_curve_plot(test_y_1,model1.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)


# In[198]:


train_1, test_1, train_y_1, test_y_1 = train_test_split(df_top10,y_resampled_2, test_size = 0.3, random_state =2) # traindata, testdata split 비율 7:3
print(train_1.shape, test_1.shape, train_y_1.shape, test_y_1.shape) # 데이터 shape 확인


# In[153]:


random_forest_model1 = RandomForestClassifier(n_estimators = 381, # 900번 추정
                                             max_features = 6, # 트리 최대 깊이 5
                                             min_samples_leaf=1,
                                             random_state = 45) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행


# In[154]:


predict1 = model1.predict(test_1) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y_1, predict1) * 100), "%") # 정확도 % 계산


# In[155]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,auc
 ##재현율과 정밀도가 비슷할수록 f1 score는 높아짐 (the harmonic mean of recall and precision)
model_metrics = {}
model_metrics['randomforest: Accuracy'] = accuracy_score(test_y_1, predict1) 
model_metrics['randomforest: Precision'] = precision_score(test_y_1, predict1)
model_metrics['randomforest: Recall'] = recall_score(test_y_1, predict1) 
model_metrics['randomforest: F1 score'] =f1_score(test_y_1, predict1) 
model_metrics['randomforest: AUC'] =roc_auc_score(test_y_1,model1.predict_proba(test_1)[:, 1])
pd.DataFrame([model_metrics])


# ### Top10- ROC커브 그리기 

# In[200]:


from sklearn.metrics import plot_roc_curve
def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    plt.figure(figsize=(10,8))
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)
    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set_facecolor((0.96, 0.96, 0.96))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate (1-Sensitivity )')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend()
    plt.show()


# In[199]:


random_forest_model1 = RandomForestClassifier(n_estimators = 381, # 900번 추정
                                             max_features = 6, # 트리 최대 깊이 5
                                            min_samples_leaf= 1,
                                             random_state = 45) # 시드값 고정
model1 = random_forest_model1.fit(train_1, train_y_1) # 학습 진행


# In[201]:


roc_curve_plot(test_y_1,model1.predict_proba(test_1)[:, 1])
#plt.savefig('rf_pet1.png', dpi = 300)

