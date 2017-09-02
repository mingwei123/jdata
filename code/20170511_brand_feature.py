# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:12:59 2017

@author: lixianpan
"""

#导包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

#加载数据
jdata_path = "E:/project/pythondata/jdata"
product = pd.read_csv(jdata_path+"/data/JData_Product.csv")
user = pd.read_csv(jdata_path+"/data/JData_User.csv",encoding='gbk')
comment = pd.read_csv(jdata_path+"/data/JData_Comment.csv")
action_201604 = pd.read_csv(jdata_path+"/data/JData_Action_201604.csv")
action_201604 = action_201604.drop_duplicates()
action_201604 = action_201604[action_201604.cate==8]

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

nowtime =datetime.now().strftime("%Y%m%d")

for i in range(3):
#    for j in range(5):
#        deltime = str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')-timedelta(j))
    j=0        
    action_201604_del = action_201604[(action_201604.time <str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')-timedelta(j)))&(action_201604.time >time[i*2])]

    brand_14_t_count = action_201604_del.brand.value_counts()
    brand_14_t_count_pd = pd.DataFrame(columns=['brand','brand_t_counts_%s'%j])
    brand_14_t_count_pd['brand'] = brand_14_t_count.index
    brand_14_t_count_pd['brand_t_counts_%s'%j] = brand_14_t_count.values

    #  1. 品牌被浏览的次数
    brand_14_t1_count = action_201604_del[(action_201604.type==1)].brand.value_counts()
    brand_14_t1_count_pd = pd.DataFrame(columns=['brand','brand_t1_counts_%s'%j])
    brand_14_t1_count_pd['brand'] = brand_14_t1_count.index
    brand_14_t1_count_pd['brand_t1_counts_%s'%j] = brand_14_t1_count.values
    
    #  2. 品牌被下单的次数
    brand_14_t4_count = action_201604_del[(action_201604.type==4)].brand.value_counts()
    brand_14_t4_count_pd = pd.DataFrame(columns=['brand','brand_t4_counts_%s'%j])
    brand_14_t4_count_pd['brand'] = brand_14_t4_count.index
    brand_14_t4_count_pd['brand_t4_counts_%s'%j] = brand_14_t4_count.values
    
    #  3. 品牌被点击的次数
    brand_14_t6_count = action_201604_del[(action_201604.type==6)].brand.value_counts()
    brand_14_t6_count_pd = pd.DataFrame(columns=['brand','brand_t6_counts_%s'%j])
    brand_14_t6_count_pd['brand'] = brand_14_t6_count.index
    brand_14_t6_count_pd['brand_t6_counts_%s'%j] = brand_14_t6_count.values
    
    #  4. 品牌被删除的次数
    brand_14_t3_count = action_201604_del[(action_201604.type==3)].brand.value_counts()
    brand_14_t3_count_pd = pd.DataFrame(columns=['brand','brand_t3_counts_%s'%j])
    brand_14_t3_count_pd['brand'] = brand_14_t3_count.index
    brand_14_t3_count_pd['brand_t3_counts_%s'%j] = brand_14_t3_count.values
    
    #  5. 品牌加购物车的次数
    brand_14_t2_count = action_201604_del[(action_201604.type==2)].brand.value_counts()
    brand_14_t2_count_pd = pd.DataFrame(columns=['brand','brand_t2_counts_%s'%j])
    brand_14_t2_count_pd['brand'] = brand_14_t2_count.index
    brand_14_t2_count_pd['brand_t2_counts_%s'%j] = brand_14_t2_count.values
    
    #  5. 品牌收藏的次数
    brand_14_t5_count = action_201604_del[(action_201604.type==5)].brand.value_counts()
    brand_14_t5_count_pd = pd.DataFrame(columns=['brand','brand_t5_counts_%s'%j])
    brand_14_t5_count_pd['brand'] = brand_14_t5_count.index
    brand_14_t5_count_pd['brand_t5_counts_%s'%j] = brand_14_t5_count.values
    #合并
    brand_14 = pd.merge(brand_14_t1_count_pd,brand_14_t4_count_pd,on='brand',how='outer')
    brand_14 = pd.merge(brand_14,brand_14_t_count_pd,on='brand',how='outer')
    brand_14 = pd.merge(brand_14,brand_14_t6_count_pd,on='brand',how='outer')
    brand_14 = pd.merge(brand_14,brand_14_t3_count_pd,on='brand',how='outer')
    brand_14 = pd.merge(brand_14,brand_14_t2_count_pd,on='brand',how='outer')
    brand_14 = pd.merge(brand_14,brand_14_t5_count_pd,on='brand',how='outer')
        
    brand_14 = brand_14.fillna(0)
    
    #  6. 被下单/被点击
    brand_14['brand_14_t4_t6_rate_%s'%j] = brand_14['brand_t4_counts_%s'%j] / (brand_14['brand_t6_counts_%s'%j]+1)
    
    #  7. 被下单/被浏览
    brand_14['brand_14_t4_t1_rate_%s'%j] = brand_14['brand_t4_counts_%s'%j] / (brand_14['brand_t1_counts_%s'%j]+1)
    brand_14['brand_14_t4_t3_rate_%s'%j] = brand_14['brand_t4_counts_%s'%j] / (brand_14['brand_t3_counts_%s'%j]+1)
        
    brand_14['brand_14_t4_t5_rate_%s'%j] = brand_14['brand_t4_counts_%s'%j] / (brand_14['brand_t5_counts_%s'%j]+1)
    #  8. 被下单/加购物车
    brand_14['brand_14_t4_t2_rate_%s'%j] = brand_14['brand_t4_counts_%s'%j] / (brand_14['brand_t2_counts_%s'%j]+1)
    
    #  9. 加购物车/被浏览
    brand_14['brand_14_t2_t1_rate_%s'%j] = brand_14['brand_t2_counts_%s'%j] / (brand_14['brand_t1_counts_%s'%j]+1)
    
    #  10. 加购物车/被点击
    brand_14['brand_14_t2_t6_rate_%s'%j] = brand_14['brand_t2_counts_%s'%j] / (brand_14['brand_t6_counts_%s'%j]+1)
    
    #  11. 加购物车/被删除
    brand_14['brand_14_t2_t3_rate_%s'%j] = brand_14['brand_t2_counts_%s'%j] / (brand_14['brand_t3_counts_%s'%j]+1)
    brand_14['brand_14_t2_t5_rate_%s'%j] = brand_14['brand_t2_counts_%s'%j] / (brand_14['brand_t5_counts_%s'%j]+1)

    brand_14.to_csv(jdata_path+'/output/%s_brand_%s_%s.csv'%(nowtime,name[i],j),index=False)
        
    brand_table = pd.read_csv(jdata_path+'/output/%s_brand_%s_0.csv'%(nowtime,name[i]))
#    brand_4 = pd.read_csv(jdata_path+'/output/%s_brand_%s_1.csv'%(nowtime,name[i]))
#    brand_3 = pd.read_csv(jdata_path+'/output/%s_brand_%s_2.csv'%(nowtime,name[i]))
#    brand_2 = pd.read_csv(jdata_path+'/output/%s_brand_%s_3.csv'%(nowtime,name[i]))
#    brand_1 = pd.read_csv(jdata_path+'/output/%s_brand_%s_4.csv'%(nowtime,name[i]))
    
#    brand_table = pd.merge(brand_1,brand_2,how='outer')
#    brand_table = pd.merge(brand_table,brand_3,how='outer')
#    brand_table = pd.merge(brand_table,brand_4,how='outer')
#    brand_table = pd.merge(brand_table,brand_5,how='outer')
    
#    brand_table['brand_t4_counts_0_1']=brand_table['brand_t4_counts_0']-brand_table['brand_t4_counts_1']
#    brand_table['brand_t4_counts_1_2']=brand_table['brand_t4_counts_1']-brand_table['brand_t4_counts_2']
#    brand_table['brand_t4_counts_2_3']=brand_table['brand_t4_counts_2']-brand_table['brand_t4_counts_3']
#    brand_table['brand_t4_counts_3_4']=brand_table['brand_t4_counts_3']-brand_table['brand_t4_counts_4']
#    brand_table['brand_t4_counts_0_1_rate'] = brand_table['brand_t4_counts_0_1']/(brand_table['brand_t4_counts_0_1'].sum())
#    brand_table['brand_t4_counts_1_2_rate'] = brand_table['brand_t4_counts_1_2']/(brand_table['brand_t4_counts_1_2'].sum())
#    brand_table['brand_t4_counts_2_3_rate'] = brand_table['brand_t4_counts_2_3']/(brand_table['brand_t4_counts_2_3'].sum())
#    brand_table['brand_t4_counts_3_4_rate'] = brand_table['brand_t4_counts_3_4']/(brand_table['brand_t4_counts_3_4'].sum())

    action_201604_1 = action_201604[action_201604.time >time[i*2]]
    action_201604_1['date'] = action_201604_1.time.apply(lambda x:x[:10])
    
    action_201604_dum = pd.get_dummies(action_201604_1.type,prefix='type')
    action_201604_dum = action_201604_1.join(action_201604_dum)
    action_201604_dum['user_id'] = action_201604_dum.user_id.astype(int)

    action_201604_cate8 = action_201604[action_201604.cate ==8]
    brand_table_user = action_201604_cate8.groupby('brand').user_id.unique().reset_index()
    brand_table_user['brand_user_count'] = brand_table_user.user_id.apply(lambda x:len(x))
    brand_table_user=brand_table_user[['brand','brand_user_count']]
    brand_table = pd.merge(brand_table,brand_table_user,on='brand',how='left') 

    brand_table_sku = action_201604_cate8.groupby('brand').sku_id.unique().reset_index()
    brand_table_sku['brand_sku_count'] = brand_table_sku.sku_id.apply(lambda x:len(x))
    brand_table_sku=brand_table_sku[['brand','brand_sku_count']]
    brand_table = pd.merge(brand_table,brand_table_sku,on='brand',how='left') 
    
    #日均行为数量
    for u in range(6):
        k=u+1
        ac1 = action_201604_dum.groupby(['brand','date']).sum()['type_%s'%k].unstack()
        ac1 =ac1.fillna(0)
        ac1_average=pd.DataFrame(ac1.mean(axis=1),columns=['brand_date_average_%s'%k]).reset_index()
        brand_table=pd.merge(brand_table,ac1_average,on='brand',how='left')
        
        a = action_201604_dum.groupby(['brand','date'],as_index=False)['type_%s'%k].sum()
        a.columns=['brand','date','type_%s_sum'%k]
        a = a.groupby('brand',as_index=False)['type_%s_sum'%k].mean()
        a.columns=['brand','type_%s_brand_date_average'%k]
        brand_table=pd.merge(brand_table,a,on='brand',how='left')
        
        a = action_201604_dum.groupby(['brand','user_id'],as_index=False)['type_%s'%k].sum()
        a.columns=['brand','user_id','type_%s_sum'%k]
        a = a.groupby('brand',as_index=False)['type_%s_sum'%k].mean()
        a.columns=['brand','type_%s_brand_user_average'%k]
        brand_table=pd.merge(brand_table,a,on='brand',how='left')


#    brand_skunum = action_201604_1.groupby('brand',as_index=False)['sku_id'].count()
#    brand_skunum.columns=['brand','brand_skunum']
##    brand_skunum_pd = pd.DataFrame(columns=['brand','brand_sku_num'])
##    brand_skunum_pd['brand'] = brand_skunum.index
##    brand_skunum_pd['brand_sku_num'] = brand_skunum.values

#    brand_table=pd.merge(brand_table,brand_skunum,on='brand',how='left')
    brand_table=brand_table.fillna(0)
    brand_table.to_csv(jdata_path+'/output/%s_brand_%s.csv'%(nowtime,name[i]),index=False)