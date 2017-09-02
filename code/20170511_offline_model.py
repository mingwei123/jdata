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


jdata_path = "E:/project/pythondata/jdata"
product = pd.read_csv(jdata_path+"/data/JData_Product.csv")
user = pd.read_csv(jdata_path+"/data/JData_User.csv")
comment = pd.read_csv(jdata_path+"/data/JData_Comment.csv")
action_201604 = pd.read_csv(jdata_path+"/data/JData_Action_201604.csv")

name=[]
name.append('train')
name.append('validate')
name.append('test')

date ='2017051016'

##读取数据
train=pd.read_csv(jdata_path+'/output/%s_feature_fuse_train.csv'%date)
test=pd.read_csv(jdata_path+'/output/%s_feature_fuse_validate.csv'%date)

train_1 =train
test_1 = test

##取出异常数据
#train_1 =train[train.type_2<7]
#test_1 = test[test.type_2<7]
#
#train_1 =train_1[train_1.type_4_sku_user_average<0.16]
#test_1 = test_1[test_1.type_4_sku_user_average<0.16]
#
#train_1 =train_1[train_1.daysum<3000]
#test_1 = test_1[test_1.daysum<3000]
#
#train_1 =train_1[train_1.user_lv_cd>2]
#test_1 = test_1[test_1.user_lv_cd>2]
#
#train_1 =train_1[train_1.type_4==0]
#test_1 = test_1[test_1.type_4==0]
#
#train_1 =train_1[train_1.u_t6_counts_1<500]
#test_1 = test_1[test_1.u_t6_counts_1<500]
#
#train_1 =train_1[train_1.type1_median<20]
#test_1 = test_1[test_1.type1_median<20]

#train_1 =train_1[train_1.hadbuy==0]
#test_1 = test_1[test_1.hadbuy==0]
##
#train_1 =train_1[train_1.age!=-1]
#test_1 = test_1[test_1.age!=-1]

##上采样
Oversampling1 = train_1.loc[train_1.hasbuy== 1]
ov =10
for i in range(ov):
    train_1 = train_1.append(Oversampling1)
    
#下采样

target = 'hasbuy'
nofeature = ['user_id','sku_id','brand','hasbuy','hasbuy']
feature = [x for x in train.columns if x not in nofeature]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

ss_train = ss.fit_transform(train_1[feature])
ss_test = ss.transform(test_1[feature])

gbdt= GradientBoostingClassifier()
gbdt.fit(ss_train,train_1[target])
predict =gbdt.predict(ss_test)


test_1['predict']=predict
test_1=test_1[test_1.predict==1]
test_predict_sort = test_1.sort(columns='type_1',ascending=False)
test_predict_sort.drop_duplicates('user_id',inplace=True)
test_predict_sort = test_predict_sort[test_predict_sort.hadbuy==0]

test_pred = pd.DataFrame(columns=['user_id','sku_id'])
test_pred['user_id'] = test_predict_sort.user_id.astype(int)
test_pred['sku_id']=test_predict_sort.sku_id.astype(int)

test_pred.to_csv(jdata_path+'/offlinesubmit/%s_gbdt_train_validation_ov%s_noyichang.csv'%(date,ov),index=False)

feature_import = pd.DataFrame(columns=['importance'],index=feature)
feature_import['importance'] =gbdt.feature_importances_
feature_import.plot(kind='barh')
plt.show()

print feature_import

print "test_pred num is:",test_pred.shape

def report(pred, label):

    actions = label
    actions['user_id']=actions.user_id.astype(int)
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
#    all_user_item_pair =  actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
#    all_user_test_item_pair =  result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print '所有用户中预测购买用户的准确率为 ' + str(all_user_acc)
    print '所有用户中预测购买用户的召回率' + str(all_user_recall)

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print '所有用户中预测购买商品的准确率为 ' + str(all_item_acc)
    print '所有用户中预测购买商品的召回率' + str(all_item_recall)
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3.0 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print 'F11=' + str(F11)
    print 'F12=' + str(F12)
    print  'ov',ov,' score=' + str(score)+" test_pred num is:",test_pred.shape
    

validate_label = pd.read_csv(jdata_path+'/output/%s_feature_fuse_validate.csv'%date)
validate_label_hasbuy = validate_label[validate_label.hasbuy ==1]
validate_label_hasbuy_us = validate_label_hasbuy[['user_id','sku_id']]
#pred = pd.read_csv(jdata_path+'/offlinesubmit/20170413_gbc_train_validation.csv')
report(test_pred,validate_label_hasbuy_us)