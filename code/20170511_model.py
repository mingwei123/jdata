# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:47:32 2017

@author: lixianpan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from datetime import datetime

jdata_path = "E:/project/pythondata/jdata"

name=[]
name.append('train')
name.append('validate')
name.append('test')

date ='2017051621'
nowtime =datetime.now().strftime("%Y%m%d%H")

##读取数据
train=pd.read_csv(jdata_path+'/output/%s_feature_fuse_validate.csv'%date)
test=pd.read_csv(jdata_path+'/output/%s_feature_fuse_test.csv'%date)

train_1 = train
test_1 = test
train_1 =train[train.type_2<7]
test_1 = test[test.type_2<7]

train_1 =train_1[train_1.type_4_sku_user_average<0.16]
test_1 = test_1[test_1.type_4_sku_user_average<0.16]

train_1 =train_1[train_1.daysum<3000]
test_1 = test_1[test_1.daysum<3000]

train_1 =train_1[train_1.user_lv_cd>2]
test_1 = test_1[test_1.user_lv_cd>2]

train_1 =train_1[train_1.type_4==0]
test_1 = test_1[test_1.type_4==0]

#train_1 =train_1[train_1.u_t6_counts_1<500]
#test_1 = test_1[test_1.u_t6_counts_1<500]

train_1 =train_1[train_1.type1_median<20]
test_1 = test_1[test_1.type1_median<20]

train_1 =train_1[train_1.hadbuy==0]
test_1 = test_1[test_1.hadbuy==0]

train_1 =train_1[train_1.age!=-1]
test_1 = test_1[test_1.age!=-1]
#train_1.to_csv(jdata_path+'/all_predict/%s_all_gbdt_train.csv'%(nowtime),index=False)

#上采样
Oversampling1 = train_1.loc[train_1.hasbuy== 1]
ov =60
for i in range(ov):
    train_1 = train_1.append(Oversampling1)
    
#下采样

target = 'hasbuy'
nofeature = ['user_id','sku_id','brand','hasbuy']
feature = [x for x in train.columns if x not in nofeature]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

ss_train = ss.fit_transform(train_1[feature])
ss_test = ss.transform(test_1[feature])


gbdt= GradientBoostingClassifier()
gbdt.fit(ss_train,train_1[target])


predict =gbdt.predict(ss_test)

#test_pred = pd.DataFrame(columns=['user_id','sku_id','predict'])
#test_pred['user_id'] = test_1.user_id.astype(int)
#test_pred['sku_id']=test_1.sku_id.astype(int)
#test_pred['predict']=predict

test_1['predict']=predict

test_1.to_csv(jdata_path+'/all_predict/%s_all_gbdt_test_validation_ov%s.csv'%(nowtime,ov),index=False)
test_1=test_1[test_1.predict==1]
test_predict_sort = test_1.sort(columns='type_1',ascending=False)
test_predict_sort.drop_duplicates('user_id',inplace=True)

test_pred = pd.DataFrame(columns=['user_id','sku_id'])
test_pred['user_id'] = test_predict_sort.user_id.astype(int)
test_pred['sku_id']=test_predict_sort.sku_id.astype(int)
test_pred[['user_id','sku_id']].to_csv(jdata_path+'/submit/%s_gbdt_test_validation_ov%s_gbc.csv'%(nowtime,ov),index=False)

feature_import = pd.DataFrame(columns=['importance'],index=feature)
feature_import['importance'] =gbdt.feature_importances_
feature_import.plot(kind='barh')
feature_import.to_csv(jdata_path+'/feature_importance/%s_gbdt_test_validation_ov%s_gbc.csv'%(nowtime,ov))
plt.show()

print feature_import

print "test_pred num is:",test_pred.shape,"ov is",ov