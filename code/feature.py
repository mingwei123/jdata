# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 21:27:28 2017

@author: lixianpan
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

jdata_path = "E:/project/pythondata/jdata"
product = pd.read_csv(jdata_path+"/data/JData_Product.csv")
user = pd.read_csv(jdata_path+"/data/JData_User.csv")
comment = pd.read_csv(jdata_path+"/data/JData_Comment.csv")
action_201604 = pd.read_csv(jdata_path+"/data/JData_Action_201604.csv")

#add user feature
def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1     
user['age'] = user['age'].map(convert_age)
del user['user_reg_tm']
#

#只保留对品类8的行为数据
action_201604 =action_201604[action_201604.cate==8]

#只保留14和15号的数据
action_201604_14_15 = action_201604[action_201604.time>'2016-04-14 00:00:00']

action_201604_14 = action_201604_14_15[action_201604_14_15.time <'2016-04-15 00:00:00']
action_201604_15 = action_201604_14_15[action_201604_14_15.time >'2016-04-15 00:00:00']

#对type取哑变量
action_201604_14_dum = pd.get_dummies(action_201604_14.type,prefix='type')
action_201604_14_dum = action_201604_14.join(action_201604_14_dum)
action_201604_15_dum = pd.get_dummies(action_201604_15.type,prefix='type')
action_201604_15_dum = action_201604_15.join(action_201604_15_dum)

#取出浏览和加购物车的最大时间
action_201604_14_dum_maxtime = action_201604_14_dum[(action_201604_14_dum.type==2)|(action_201604_14_dum.type==6)].groupby(['user_id','sku_id'],as_index=False)['time'].max()
action_201604_15_dum_maxtime = action_201604_15_dum[(action_201604_15_dum.type==2)|(action_201604_15_dum.type==6)].groupby(['user_id','sku_id'],as_index=False)['time'].max()

#sku_id的总下单次数
sku_count_14 = action_201604[(action_201604.type==4) &(action_201604.time <'2016-04-15 00:00:00')].sku_id.value_counts()
sku_count_14_pd = pd.DataFrame(columns=['sku_id','t4_counts'])
sku_count_14_pd['sku_id'] = sku_count_14.index
sku_count_14_pd['t4_counts'] = sku_count_14.values

sku_count_15 = action_201604[(action_201604.type==4)].sku_id.value_counts()
sku_count_15_pd = pd.DataFrame(columns=['sku_id','t4_counts'])
sku_count_15_pd['sku_id'] = sku_count_15.index
sku_count_15_pd['t4_counts'] = sku_count_15.values

#user_id的总下单次数
user_count_14 =action_201604[(action_201604.type==4) &(action_201604.time <'2016-04-15 00:00:00')].user_id.value_counts()
user_count_14_pd = pd.DataFrame(columns=['user_id','u_t4_counts'])
user_count_14_pd['user_id'] = user_count_14.index
user_count_14_pd['u_t4_counts'] = user_count_14.values

user_count_15 = action_201604[(action_201604.type==4) ].user_id.value_counts()
user_count_15_pd = pd.DataFrame(columns=['user_id','u_t4_counts'])
user_count_15_pd['user_id'] = user_count_15.index
user_count_15_pd['u_t4_counts'] = user_count_15.values

#sku_id的总浏览次数
sku_count_14_t6 = action_201604[(action_201604.type==6) &(action_201604.time <'2016-04-15 00:00:00')].sku_id.value_counts()
sku_count_14_t6_pd = pd.DataFrame(columns=['sku_id','s_t6_counts'])
sku_count_14_t6_pd['sku_id'] = sku_count_14_t6.index
sku_count_14_t6_pd['s_t6_counts'] = sku_count_14_t6.values

sku_count_15_t6 = action_201604[(action_201604.type==6) ].sku_id.value_counts()
sku_count_15_t6_pd = pd.DataFrame(columns=['sku_id','s_t6_counts'])
sku_count_15_t6_pd['sku_id'] = sku_count_15_t6.index
sku_count_15_t6_pd['s_t6_counts'] = sku_count_15_t6.values

#user_id的总浏览次数
user_count_14_t6 =action_201604[(action_201604.type==6) &(action_201604.time <'2016-04-15 00:00:00')].user_id.value_counts()
user_count_14_t6_pd = pd.DataFrame(columns=['user_id','u_t6_counts'])
user_count_14_t6_pd['user_id'] = user_count_14_t6.index
user_count_14_t6_pd['u_t6_counts'] = user_count_14_t6.values

user_count_15_t6 = action_201604[(action_201604.type==6) ].user_id.value_counts()
user_count_15_t6_pd = pd.DataFrame(columns=['user_id','u_t6_counts'])
user_count_15_t6_pd['user_id'] = user_count_15_t6.index
user_count_15_t6_pd['u_t6_counts'] = user_count_15_t6.values

#type1-6的统计
action_201604_14_dum_gb = action_201604_14_dum.groupby(['user_id','sku_id'],as_index=False)['type_1','type_2','type_3','type_4','type_5','type_6'].sum()
action_201604_15_dum_gb = action_201604_15_dum.groupby(['user_id','sku_id'],as_index=False)['type_1','type_2','type_3','type_4','type_5','type_6'].sum()

#第二次修改：
action_201604_14_dum_gb = action_201604_14_dum_gb[action_201604_14_dum_gb.type_4 ==0]
action_201604_15_dum_gb = action_201604_15_dum_gb[action_201604_15_dum_gb.type_4 ==0]

#取出15号下单的数据
action_201604_15_t4 = action_201604_15[action_201604_15.type==4]
action_201604_15_t4['hasbuy']=1
action_201604_15_t4_label = action_201604_15_t4[['user_id','sku_id','hasbuy']]

#开始合并特征
train = pd.merge(action_201604_14_dum_gb,action_201604_14_dum_maxtime,on=['user_id','sku_id'])
test=pd.merge(action_201604_15_dum_gb ,action_201604_15_dum_maxtime,on=['user_id','sku_id'])

train  = pd.merge(train,sku_count_14_pd,on='sku_id',how='left')
test = pd.merge(test,sku_count_15_pd,on='sku_id',how='left')

train = pd.merge(train,user_count_14_pd,on='user_id',how='left')
test = pd.merge(test,user_count_15_pd,on='user_id',how='left')

train = pd.merge(train,sku_count_14_t6_pd ,on='sku_id',how='left')
test = pd.merge(test,sku_count_15_t6_pd ,on='sku_id',how='left')

train = pd.merge(train,user_count_14_t6_pd,on='user_id',how='left')
test = pd.merge(test,user_count_15_t6_pd,on='user_id',how='left')

#加入train 的标签
train_1 = pd.merge(train,action_201604_15_t4_label ,on=['user_id','sku_id'],how='left')

#保存一个版本
train_1.to_csv(jdata_path+'/output/20170405_train.csv',index=False)
test.to_csv(jdata_path+'/output/20170405_test.csv',index=False)

del train_1['time']
del test['time']

#add feature merge
train_1 = pd.merge(train_1,user,on='user_id')
test = pd.merge(test,user,on='user_id')
#

train = train_1.fillna(0)
test = test.fillna(0)

train_1 = train[train.type_2 >0]
test_1 = test[test.type_2 >0]

Oversampling1 = train_1.loc[train_1.hasbuy== 1]
for i in range(250):
    train_1 = train_1.append(Oversampling1)
    
target = 'hasbuy'
feature = [x for x in train.columns if x not in [target]]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

ss_train = ss.fit_transform(train_1[feature])
ss_test = ss.transform(test_1)

gbc = XGBClassifier()
gbc.fit(ss_train,train_1[target])
predict =gbc.predict(ss_test)

test_pred = pd.DataFrame(columns=['user_id','sku_id','predict'])
test_pred['user_id'] = test_1.user_id
test_pred['sku_id']=test_1.sku_id
test_pred['predict']=predict

act04 = test_pred[test_pred.predict==1]
act_04 = act04.drop_duplicates('user_id')
#act_04[['user_id','sku_id']].to_csv(jdata_path+'/submit/20170406_gbc_1.csv',index=False)


#type2=0.type1>1
train_2 = train[train.type_2 ==0]
test_2 = test[test.type_2 ==0]

Oversampling1 = train_2.loc[train_2.hasbuy== 1]
for i in range(250):
    train_2 = train_2.append(Oversampling1)
    
target = 'hasbuy'
feature = [x for x in train.columns if x not in [target]]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

ss_train = ss.fit_transform(train_2[feature])
ss_test = ss.transform(test_2)

gbc = XGBClassifier()
gbc.fit(ss_train,train_2[target])
predict =gbc.predict(ss_test)

test_pred_2 = pd.DataFrame(columns=['user_id','sku_id','predict'])
test_pred_2['user_id'] = test_2.user_id
test_pred_2['sku_id']=test_2.sku_id
test_pred_2['predict']=predict

act04_2 = test_pred_2[test_pred_2.predict==1]
act_04_2 = act04_2.drop_duplicates('user_id')
#act_04_2[['user_id','sku_id']].to_csv(jdata_path+'/submit/20170406_gbc_2.csv',index=False)

act_04_0 = pd.concat([act_04,act_04_2])
act_04=act_04_0.drop_duplicates('user_id')
act_04[['user_id','sku_id']].to_csv(jdata_path+'/submit/20170406_gbc_user_int.csv',index=False)

print "the length is:",act04.shape