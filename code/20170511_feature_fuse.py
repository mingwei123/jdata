# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:11:43 2017

@author: lixianpan
"""

import pandas as pd
import numpy as np
from datetime import datetime


jdata_path = "E:/project/pythondata/jdata"
#product = pd.read_csv(jdata_path+"/data/JData_Product.csv")
#user = pd.read_csv(jdata_path+"/data/JData_User.csv")
#comment = pd.read_csv(jdata_path+"/data/JData_Comment.csv")
action_201604 = pd.read_csv(jdata_path+"/data/JData_Action_201604.csv")

train_start_time = '2016-04-01 00:00:00'
train_end_time = '2016-04-06 00:00:00'
validate_start_time =train_end_time
validate_end_time = '2016-04-11 00:00:00'
test_start_time = validate_end_time
test_end_time = '2016-04-16 00:00:00'

time =[]
time.append(train_start_time)
time.append(train_end_time)
time.append(validate_start_time)
time.append(validate_end_time)
time.append(test_start_time)
time.append(test_end_time)

name=[]
name.append('train')
name.append('validate')
name.append('test')

date ='20170516'

nowtime =datetime.now().strftime("%Y%m%d%H")

    
for i in range(3):

    user_14 = pd.read_csv(jdata_path+'/output/%s_user_%s.csv'%(date,name[i]))
    
    sku_14 = pd.read_csv(jdata_path+'/output/%s_sku_%s.csv'%(date,name[i]))
    
    brand_14 = pd.read_csv(jdata_path+'/output/%s_brand_%s.csv'%(date,name[i]))
    
    comment_14 = pd.read_csv(jdata_path+'/output/%s_comment_%s.csv'%(date,name[i]))
    
    #只保留对品类8的行为数据
    action_201604 =action_201604[action_201604.cate==8]
      
    action_201604_14 = action_201604[(action_201604.time <time[2*i+1])&(action_201604.time > time[2*i])]
    
    #对type取哑变量
    action_201604_14_dum = pd.get_dummies(action_201604_14.type,prefix='type')
    action_201604_14_dum = action_201604_14.join(action_201604_14_dum)
    
    #type1-6的统计
    action_201604_14_dum_gb_t2_t3_t4 = action_201604_14_dum.groupby(['user_id','sku_id'],as_index=False)['type_1','type_2','type_3','type_4','type_5','type_6'].sum()
    action_201604_14_dum_gb_t1 = action_201604_14_dum_gb_t2_t3_t4.groupby('user_id',as_index=False)['type_1'].agg({'type1_mean':np.mean,'type1_median':np.median,'type1_max':np.max,'type1_min':np.min,'type1_std':np.std})
    action_201604_14_dum_gb_t6 = action_201604_14_dum_gb_t2_t3_t4.groupby('user_id',as_index=False)['type_6'].agg({'type6_mean':np.mean,'type6_median':np.median,'type6_max':np.max,'type6_min':np.min,'type6_std':np.std})
    action_201604_14_dum_gb=pd.merge(action_201604_14_dum_gb_t1,action_201604_14_dum_gb_t6,on='user_id')  
    
    action_201604_14_dum_gb_user_sku_t1 = action_201604_14_dum_gb_t2_t3_t4.groupby(['user_id','sku_id'],as_index=False)['type_1'].agg({'type1_sku_mean':np.mean,'type1_sku_sum':np.sum,'type1_sku_median':np.median,'type1_sku_max':np.max,'type1_sku_min':np.min,'type1_sku_std':np.std})
    action_201604_14_dum_gb_user_sku_t6 = action_201604_14_dum_gb_t2_t3_t4.groupby(['user_id','sku_id'],as_index=False)['type_6'].agg({'type6_sku_mean':np.mean,'type6_sku_sum':np.sum,'type6_sku_median':np.median,'type6_sku_max':np.max,'type6_sku_min':np.min,'type6_sku_std':np.std})
    action_201604_14_dum_gb_user_sku=pd.merge(action_201604_14_dum_gb_user_sku_t1,action_201604_14_dum_gb_user_sku_t6,on=['user_id','sku_id'])
    
    action_201604_14_dum['weekday'] = pd.Index(pd.to_datetime(action_201604_14_dum.time)).weekday+1
    action_201604_14_dum.sort(columns='weekday',ascending=False)
    action_201604_14_dum.drop_duplicates(['user_id','sku_id'],inplace=True)
    
    train = pd.merge(action_201604_14_dum_gb_t2_t3_t4,user_14,on='user_id',how='outer')
    
    train = pd.merge(train,action_201604_14_dum_gb,on='user_id',how='outer')
    
    train = pd.merge(train,sku_14,on='sku_id',how='left')
    
#    train = pd.merge(train,product,on='sku_id',how='left')
    
    train = pd.merge(train,comment_14,on='sku_id',how='left')
    
    action_201604_14_dum_week =action_201604_14_dum[['user_id','sku_id','weekday']]
    train = pd.merge(train,action_201604_14_dum_week,on=['user_id','sku_id'],how='left')
    
    train = pd.merge(train,action_201604_14_dum_gb_user_sku,on=['user_id','sku_id'],how='left')
    
#    train.to_csv(jdata_path+'/output/%s_feature_fuse_%s_temp.csv'%(nowtime,name[i]),index=False)
    train = pd.merge(train,brand_14,on='brand',how='left')
    
    if i<2:
    #取出15号下单的数据
        action_201604_15_t4 = action_201604[(action_201604.type==4)&(action_201604.time <time[2*i+3])&(action_201604.time > time[2*i+2])]
        action_201604_15_t4.loc[:,'hasbuy']=1
        action_201604_15_t4_label = action_201604_15_t4[['user_id','sku_id','hasbuy']]
        train = pd.merge(train,action_201604_15_t4_label ,on=['user_id','sku_id'],how='left')
    
    train = train.fillna(0)
    
#    del train['user_reg_tm']
    #保存一个版本
    train.to_csv(jdata_path+'/output/%s_feature_fuse_%s.csv'%(nowtime,name[i]),index=False)