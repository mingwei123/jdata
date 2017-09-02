# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:39:43 2017

@author: lixianpan
"""

#导包
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

#加载数据
jdata_path = "E:/project/pythondata/jdata"
user = pd.read_csv(jdata_path+"/data/JData_User.csv",encoding='gbk')
action_201604 = pd.read_csv(jdata_path+"/data/JData_Action_201604.csv")
action_201603 = pd.read_csv(jdata_path+"/data/JData_Action_201603.csv")
action_201602 = pd.read_csv(jdata_path+"/data/JData_Action_201602.csv")

action_201604=action_201604.drop_duplicates()
action_201603=action_201603.drop_duplicates()
action_201602=action_201602.drop_duplicates()

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

#年龄映射
def convert_age(age_str):
    if age_str == u'-1':
        return 1
    elif age_str == u'15岁以下':
        return 2
    elif age_str == u'16-25岁':
        return 3
    elif age_str == u'26-35岁':
        return 4
    elif age_str == u'36-45岁':
        return 5
    elif age_str == u'46-55岁':
        return 6
    elif age_str == u'56岁以上':
        return 7
    else:
        return 0

#注册时间的处理
user.user_reg_tm.fillna('2016-04-15',inplace=True)
user['daysum'] = user.user_reg_tm.apply(lambda x: (datetime.strptime('2016-04-16','%Y-%m-%d') - datetime.strptime(str(x),'%Y-%m-%d')).days)
user['day_lv_rate'] =user['daysum'] /user['user_lv_cd']
    
#user_sex_dum = pd.get_dummies(user.sex,prefix='sex')
#user = user.join(user_sex_dum)
    
user['age'] = user['age'].map(convert_age)
#age_df = pd.get_dummies(user["age"], prefix="age")
#user = user.join(age_df)

nowtime =datetime.now().strftime("%Y%m%d")

for i in range(3):
    #j 用5天减去j
#    for j in range(5):
        #User -timedelta(j)
    k=5
        
    user_14_t_counts =action_201604[(action_201604.time <str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')))&(action_201604.time >time[i*2])].user_id.value_counts()
    user_14_t_counts_pd = pd.DataFrame(columns=['user_id','u_t_counts_%s'%k])
    user_14_t_counts_pd['user_id'] = user_14_t_counts.index
    user_14_t_counts_pd['u_t_counts_%s'%k] = user_14_t_counts.values

    user_14_t1_counts =action_201604[(action_201604.type==1) &(action_201604.time <str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')))&(action_201604.time >time[i*2])].user_id.value_counts()
    user_14_t1_counts_pd = pd.DataFrame(columns=['user_id','u_t1_counts_%s'%k])
    user_14_t1_counts_pd['user_id'] = user_14_t1_counts.index
    user_14_t1_counts_pd['u_t1_counts_%s'%k] = user_14_t1_counts.values

    user_14_t2_counts =action_201604[(action_201604.type==2) &(action_201604.time <str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')))&(action_201604.time >time[i*2])].user_id.value_counts()
    user_14_t2_counts_pd = pd.DataFrame(columns=['user_id','u_t2_counts_%s'%k])
    user_14_t2_counts_pd['user_id'] = user_14_t2_counts.index
    user_14_t2_counts_pd['u_t2_counts_%s'%k] = user_14_t2_counts.values

    user_14_t3_counts =action_201604[(action_201604.type==3) &(action_201604.time <str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')))&(action_201604.time >time[i*2])].user_id.value_counts()
    user_14_t3_counts_pd = pd.DataFrame(columns=['user_id','u_t3_counts_%s'%k])
    user_14_t3_counts_pd['user_id'] = user_14_t3_counts.index
    user_14_t3_counts_pd['u_t3_counts_%s'%k] = user_14_t3_counts.values
    
    user_14_t4_counts =action_201604[(action_201604.type==4) &(action_201604.time <str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')))&(action_201604.time >time[i*2])].user_id.value_counts()
    user_14_t4_counts_pd = pd.DataFrame(columns=['user_id','u_t4_counts_%s'%k])
    user_14_t4_counts_pd['user_id'] = user_14_t4_counts.index
    user_14_t4_counts_pd['u_t4_counts_%s'%k] = user_14_t4_counts.values

    user_14_t5_counts =action_201604[(action_201604.type==5) &(action_201604.time <str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')))&(action_201604.time >time[i*2])].user_id.value_counts()
    user_14_t5_counts_pd = pd.DataFrame(columns=['user_id','u_t5_counts_%s'%k])
    user_14_t5_counts_pd['user_id'] = user_14_t5_counts.index
    user_14_t5_counts_pd['u_t5_counts_%s'%k] = user_14_t5_counts.values
    
    user_14_t6_counts =action_201604[(action_201604.type==6) &(action_201604.time < str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')))&(action_201604.time >time[i*2])].user_id.value_counts()
    user_14_t6_counts_pd = pd.DataFrame(columns=['user_id','u_t6_counts_%s'%k])
    user_14_t6_counts_pd['user_id'] = user_14_t6_counts.index
    user_14_t6_counts_pd['u_t6_counts_%s'%k] = user_14_t6_counts.values
    
    user_14_t6_216_counts =action_201604[(action_201604.type==6)&(action_201604.time <str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')))&(action_201604.time >time[i*2])&(action_201604.model_id == 216)].user_id.value_counts()
    user_14_t6_216_counts_pd = pd.DataFrame(columns=['user_id','u_t6_216_counts_%s'%k])
    user_14_t6_216_counts_pd['user_id'] = user_14_t6_216_counts.index
    user_14_t6_216_counts_pd['u_t6_216_counts_%s'%k] = user_14_t6_216_counts.values
    
    user_14_t6_217_counts =action_201604[(action_201604.type==6) &(action_201604.time <str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')))&(action_201604.time >time[i*2])&(action_201604.model_id == 217)].user_id.value_counts()
    user_14_t6_217_counts_pd = pd.DataFrame(columns=['user_id','u_t6_217_counts_%s'%k])
    user_14_t6_217_counts_pd['user_id'] = user_14_t6_217_counts.index
    user_14_t6_217_counts_pd['u_t6_217_counts_%s'%k] = user_14_t6_217_counts.values

    user_04_hadbuy_cate8 = action_201604[(action_201604.type==4)&(action_201604.cate ==8)&(action_201604.time <time[i*2])].user_id
    user_03_hadbuy_cate8 = action_201603[(action_201603.type==4)&(action_201603.cate ==8)].user_id
    user_02_hadbuy_cate8 = action_201602[(action_201602.type==4)&(action_201602.cate ==8)].user_id
    user_14_hadbuy_cate8 =pd.concat([user_04_hadbuy_cate8,user_03_hadbuy_cate8,user_02_hadbuy_cate8])
    
    user_14_04_hadbuy_cate8_pd =pd.DataFrame(columns=['user_id','hadbuy'])
    user_14_04_hadbuy_cate8_pd['user_id'] = user_14_hadbuy_cate8
    user_14_04_hadbuy_cate8_pd=user_14_04_hadbuy_cate8_pd.drop_duplicates('user_id')
    user_14_04_hadbuy_cate8_pd['hadbuy'] = 1
    
        #合并User表
    user_14 = pd.merge(user,user_14_t_counts_pd,on='user_id',how='left')
    user_14 = pd.merge(user_14,user_14_t1_counts_pd,on='user_id',how='left')
    user_14 = pd.merge(user_14,user_14_t2_counts_pd,on='user_id',how='left')
    user_14 = pd.merge(user_14,user_14_t3_counts_pd,on='user_id',how='left')
    user_14 = pd.merge(user_14,user_14_t4_counts_pd,on='user_id',how='left')
    user_14 = pd.merge(user_14,user_14_t5_counts_pd,on='user_id',how='left')
    user_14 = pd.merge(user_14,user_14_t6_counts_pd,on='user_id',how='left')
    user_14 = pd.merge(user_14,user_14_t6_216_counts_pd,on='user_id',how='left')
    user_14 = pd.merge(user_14,user_14_t6_217_counts_pd,on='user_id',how='left')
    user_14 = pd.merge(user_14,user_14_04_hadbuy_cate8_pd,on='user_id',how='left')

    user_14 = user_14.fillna(0)
        
    user_14['u_t_t1_rate_%s'%k] = (user_14['u_t1_counts_%s'%k])/(user_14['u_t_counts_%s'%k]+1)
    user_14['u_t_t2_rate_%s'%k] = (user_14['u_t2_counts_%s'%k])/(user_14['u_t_counts_%s'%k]+1)
    user_14['u_t_t3_rate_%s'%k] = (user_14['u_t3_counts_%s'%k])/(user_14['u_t_counts_%s'%k]+1)
    user_14['u_t_t4_rate_%s'%k] = (user_14['u_t4_counts_%s'%k])/(user_14['u_t_counts_%s'%k]+1)
    user_14['u_t_t5_rate_%s'%k] = (user_14['u_t5_counts_%s'%k])/(user_14['u_t_counts_%s'%k]+1)
    user_14['u_t_t6_rate_%s'%k] = (user_14['u_t6_counts_%s'%k])/(user_14['u_t_counts_%s'%k]+1)

    user_14['u_t4_t1_rate_%s'%k] = (user_14['u_t4_counts_%s'%k]) / (user_14['u_t1_counts_%s'%k]+1)
    user_14['u_t4_t2_rate_%s'%k] = (user_14['u_t4_counts_%s'%k]) / (user_14['u_t2_counts_%s'%k]+1)
    user_14['u_t4_t3_rate_%s'%k] = (user_14['u_t4_counts_%s'%k]) / (user_14['u_t3_counts_%s'%k]+1)
    user_14['u_t4_t5_rate_%s'%k] = (user_14['u_t4_counts_%s'%k]) / (user_14['u_t5_counts_%s'%k]+1)
    user_14['u_t4_t6_rate_%s'%k] = (user_14['u_t4_counts_%s'%k]) / (user_14['u_t6_counts_%s'%k]+1)
        
    user_14['u_t2_t1_rate_%s'%k] = (user_14['u_t2_counts_%s'%k]) / (user_14['u_t1_counts_%s'%k]+1)
    user_14['u_t2_t3_rate_%s'%k] = (user_14['u_t2_counts_%s'%k]) / (user_14['u_t3_counts_%s'%k]+1)    
    user_14['u_t2_t5_rate_%s'%k] = (user_14['u_t2_counts_%s'%k]) / (user_14['u_t5_counts_%s'%k]+1)
    user_14['u_t2_t6_rate_%s'%k] = (user_14['u_t2_counts_%s'%k]) / (user_14['u_t6_counts_%s'%k]+1)
    
    user_14.to_csv(jdata_path+'/output/%s_user_%s_fillna_%s.csv'%(nowtime,name[i],k),index=False)
    user_1 = pd.read_csv(jdata_path+'/output/%s_user_%s_fillna_5.csv'%(nowtime,name[i]))
#    user_2 = pd.read_csv(jdata_path+'/output/%s_user_%s_fillna_1.csv'%(nowtime,name[i]))
#    user_3 = pd.read_csv(jdata_path+'/output/%s_user_%s_fillna_2.csv'%(nowtime,name[i]))
#    user_4 = pd.read_csv(jdata_path+'/output/%s_user_%s_fillna_3.csv'%(nowtime,name[i]))
#    user_5 = pd.read_csv(jdata_path+'/output/%s_user_%s_fillna_4.csv'%(nowtime,name[i]))
    
    user_table = pd.merge(user,user_1,how='left')
#    user_table = pd.merge(user_table,user_2,how='left')
#    user_table = pd.merge(user_table,user_3,how='left')
#    user_table = pd.merge(user_table,user_4,how='left')
#    user_table = pd.merge(user_table,user_5,how='left')
    
    action_201604_cate8 = action_201604[action_201604.cate ==8]
    user_table_sku = action_201604_cate8.groupby('user_id').sku_id.unique().reset_index()
    user_table_sku['user_sku_count'] = user_table_sku.sku_id.apply(lambda x:len(x))
    user_table_sku=user_table_sku[['user_id','user_sku_count']]
    user_table = pd.merge(user_table,user_table_sku,how='left') 
    
    user_table_brand = action_201604_cate8.groupby('user_id').brand.unique().reset_index()
    user_table_brand['user_brand_count'] = user_table_brand.brand.apply(lambda x:len(x))
    user_table_brand=user_table_brand[['user_id','user_brand_count']]
    user_table = pd.merge(user_table,user_table_brand,how='left')   
        
    #活跃天数
    action_201604_del=action_201604[(action_201604.time < time[i*2+1])&(action_201604.time >time[i*2])]
    action_201604_del['date'] = action_201604_del.time.apply(lambda x:x[:10])
    action_201604_active = action_201604_del.drop_duplicates(['user_id','date'])
    action_201604_active = action_201604_active.date.value_counts()
    action_201604_active_pd = pd.DataFrame(columns=['user_id','active_day'])
    action_201604_active_pd['user_id'] = action_201604_active.index
    action_201604_active_pd['active_day'] = action_201604_active.values
    user_table = pd.merge(user_table,action_201604_active_pd,how='left')
    
    #最近时间
    action_201604_del_dropbytime = action_201604_del.sort(columns='time',ascending=False)
    action_201604_del_dropbytime =action_201604_del_dropbytime.drop_duplicates('time')
    action_201604_del_dropbytime['long_time'] = action_201604_del_dropbytime.time.apply(lambda x: (datetime.strptime('2016-04-16 00:00:00','%Y-%m-%d %H:%M:%S') - datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S')).seconds)
    action_201604_del_dropbytime['long_time_hour'] = action_201604_del_dropbytime['long_time']/60.0
    action_201604_del_dropbytime_pd = action_201604_del_dropbytime[['user_id','long_time','long_time_hour']]
    user_table = pd.merge(user_table,action_201604_del_dropbytime_pd,how='left')
    

    user_table = user_table.drop_duplicates('user_id')
    user_table=user_table.fillna(0)
    
    del user_table['user_reg_tm']
    user_table.to_csv(jdata_path+'/output/%s_user_%s.csv'%(nowtime,name[i]),index=False)
    