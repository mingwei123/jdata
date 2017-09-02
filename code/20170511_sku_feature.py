# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:58:00 2017

@author: lixianpan
"""
#导包
import pandas as pd
import numpy as np
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
    #j 用5天减去j
#    for j in range(5):
#        deltime = str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')-timedelta(j)) -timedelta(j)
    j=0
    action_201604_del = action_201604[(action_201604.time <str(datetime.strptime(time[i*2+1],'%Y-%m-%d %H:%M:%S')))&(action_201604.time >time[i*2])]
    sku_14_t_counts =action_201604_del.sku_id.value_counts()
    sku_14_t_counts_pd = pd.DataFrame(columns=['sku_id','sku_t_counts_%s'%j])
    sku_14_t_counts_pd['sku_id'] = sku_14_t_counts.index
    sku_14_t_counts_pd['sku_t_counts_%s'%j] = sku_14_t_counts.values

    #  1. 商品被浏览的次数
    sku_14_t1_count = action_201604_del[(action_201604.type==1)].sku_id.value_counts()
    sku_14_t1_count_pd = pd.DataFrame(columns=['sku_id','sku_t1_counts_%s'%j])
    sku_14_t1_count_pd['sku_id'] = sku_14_t1_count.index
    sku_14_t1_count_pd['sku_t1_counts_%s'%j] = sku_14_t1_count.values

     #  2. 商品被加购物车的次数
    sku_14_t2_count = action_201604_del[(action_201604.type==2)].sku_id.value_counts()
    sku_14_t2_count_pd = pd.DataFrame(columns=['sku_id','sku_t2_counts_%s'%j])
    sku_14_t2_count_pd['sku_id'] = sku_14_t2_count.index
    sku_14_t2_count_pd['sku_t2_counts_%s'%j] = sku_14_t2_count.values
    
    #  3. 商品被删除的次数
    sku_14_t3_count = action_201604_del[(action_201604.type==3)].sku_id.value_counts()
    sku_14_t3_count_pd = pd.DataFrame(columns=['sku_id','sku_t3_counts_%s'%j])
    sku_14_t3_count_pd['sku_id'] = sku_14_t3_count.index
    sku_14_t3_count_pd['sku_t3_counts_%s'%j] = sku_14_t3_count.values
    
    #  4. 商品被下单的次数
    sku_14_t4_count = action_201604_del[(action_201604.type==4)].sku_id.value_counts()
    sku_14_t4_count_pd = pd.DataFrame(columns=['sku_id','sku_t4_counts_%s'%j])
    sku_14_t4_count_pd['sku_id'] = sku_14_t4_count.index
    sku_14_t4_count_pd['sku_t4_counts_%s'%j] = sku_14_t4_count.values

    #  5. 商品被收藏的次数
    sku_14_t5_count = action_201604_del[(action_201604.type==5)].sku_id.value_counts()
    sku_14_t5_count_pd = pd.DataFrame(columns=['sku_id','sku_t5_counts_%s'%j])
    sku_14_t5_count_pd['sku_id'] = sku_14_t5_count.index
    sku_14_t5_count_pd['sku_t5_counts_%s'%j] = sku_14_t5_count.values
   
    #  6. 商品被点击的次数
    sku_14_t6_count = action_201604_del[(action_201604.type==6)].sku_id.value_counts()
    sku_14_t6_count_pd = pd.DataFrame(columns=['sku_id','sku_t6_counts_%s'%j])
    sku_14_t6_count_pd['sku_id'] = sku_14_t6_count.index
    sku_14_t6_count_pd['sku_t6_counts_%s'%j] = sku_14_t6_count.values

    #  7. 商品被点击的次数_model_id 216
    sku_14_t6_count_216 = action_201604_del[(action_201604.type==6)&(action_201604.model_id==216)].sku_id.value_counts()
    sku_14_t6_count_216_pd = pd.DataFrame(columns=['sku_id','sku_t6_counts_216_%s'%j])
    sku_14_t6_count_216_pd['sku_id'] = sku_14_t6_count_216.index
    sku_14_t6_count_216_pd['sku_t6_counts_216_%s'%j] = sku_14_t6_count_216.values

    #  8. 商品被点击的次数_model_id 217
    sku_14_t6_count_217 = action_201604_del[(action_201604.type==6)&(action_201604.model_id==217)].sku_id.value_counts()
    sku_14_t6_count_217_pd = pd.DataFrame(columns=['sku_id','sku_t6_counts_217_%s'%j])
    sku_14_t6_count_217_pd['sku_id'] = sku_14_t6_count_217.index
    sku_14_t6_count_217_pd['sku_t6_counts_217_%s'%j] = sku_14_t6_count_217.values

    #  9. 商品被点击的次数_model_id 26
    sku_14_t6_count_26 = action_201604_del[(action_201604.type==6)&(action_201604.model_id==26)].sku_id.value_counts()
    sku_14_t6_count_26_pd = pd.DataFrame(columns=['sku_id','sku_t6_counts_26_%s'%j])
    sku_14_t6_count_26_pd['sku_id'] = sku_14_t6_count_26.index
    sku_14_t6_count_26_pd['sku_t6_counts_26_%s'%j] = sku_14_t6_count_26.values

    #  10. 商品被点击的次数_model_id 27
    sku_14_t6_count_27 = action_201604_del[(action_201604.type==6)&(action_201604.model_id==27)].sku_id.value_counts()
    sku_14_t6_count_27_pd = pd.DataFrame(columns=['sku_id','sku_t6_counts_27_%s'%j])
    sku_14_t6_count_27_pd['sku_id'] = sku_14_t6_count_27.index
    sku_14_t6_count_27_pd['sku_t6_counts_27_%s'%j] = sku_14_t6_count_27.values
    #合并：
    sku_14 = pd.merge(sku_14_t2_count_pd,sku_14_t1_count_pd,on='sku_id',how='outer')
    sku_14 = pd.merge(sku_14,sku_14_t_counts_pd,on='sku_id',how='outer')
    sku_14 = pd.merge(sku_14 ,sku_14_t4_count_pd,on='sku_id',how='outer')
    sku_14 = pd.merge(sku_14 ,sku_14_t6_count_pd,on='sku_id',how='outer')
    sku_14 = pd.merge(sku_14 ,sku_14_t3_count_pd,on='sku_id',how='outer')
    sku_14 = pd.merge(sku_14 ,sku_14_t5_count_pd,on='sku_id',how='outer')
    sku_14 = pd.merge(sku_14 ,sku_14_t6_count_216_pd,on='sku_id',how='outer')
    sku_14 = pd.merge(sku_14 ,sku_14_t6_count_217_pd,on='sku_id',how='outer')        
    sku_14 = pd.merge(sku_14 ,sku_14_t6_count_26_pd,on='sku_id',how='outer')  
    sku_14 = pd.merge(sku_14 ,sku_14_t6_count_27_pd,on='sku_id',how='outer')  
    sku_14 = sku_14.fillna(0)

    sku_14['sku_t_t1_rate_%s'%j] = (sku_14['sku_t1_counts_%s'%j])/(sku_14['sku_t_counts_%s'%j]+1)
    sku_14['sku_t_t2_rate_%s'%j] = (sku_14['sku_t2_counts_%s'%j])/(sku_14['sku_t_counts_%s'%j]+1)
    sku_14['sku_t_t3_rate_%s'%j] = (sku_14['sku_t3_counts_%s'%j])/(sku_14['sku_t_counts_%s'%j]+1)
    sku_14['sku_t_t4_rate_%s'%j] = (sku_14['sku_t4_counts_%s'%j])/(sku_14['sku_t_counts_%s'%j]+1)
    sku_14['sku_t_t5_rate_%s'%j] = (sku_14['sku_t5_counts_%s'%j])/(sku_14['sku_t_counts_%s'%j]+1)
    sku_14['sku_t_t6_rate_%s'%j] = (sku_14['sku_t6_counts_%s'%j])/(sku_14['sku_t_counts_%s'%j]+1)
        
    #  6. 下单/商品加购物车
    sku_14['sku_14_t4_t2_rate_%s'%j] = sku_14['sku_t4_counts_%s'%j] / (sku_14['sku_t2_counts_%s'%j]+1)
    sku_14['sku_14_t4_t1_rate_%s'%j] = sku_14['sku_t4_counts_%s'%j] / (sku_14['sku_t1_counts_%s'%j]+1)
    sku_14['sku_14_t4_t6_rate_%s'%j] = sku_14['sku_t4_counts_%s'%j] / (sku_14['sku_t6_counts_%s'%j]+1)
    sku_14['sku_14_t4_t3_rate_%s'%j] = sku_14['sku_t4_counts_%s'%j] / (sku_14['sku_t3_counts_%s'%j]+1)
    sku_14['sku_14_t4_t5_rate_%s'%j] = sku_14['sku_t4_counts_%s'%j] / (sku_14['sku_t5_counts_%s'%j]+1)
    sku_14['sku_14_t4_216_rate_%s'%j] = sku_14['sku_t4_counts_%s'%j] / (sku_14['sku_t6_counts_216_%s'%j]+1)
    sku_14['sku_14_t4_217_rate_%s'%j] = sku_14['sku_t4_counts_%s'%j] / (sku_14['sku_t6_counts_217_%s'%j]+1)
    sku_14['sku_14_t4_26_rate_%s'%j] = sku_14['sku_t4_counts_%s'%j] / (sku_14['sku_t6_counts_26_%s'%j]+1)
    sku_14['sku_14_t4_27_rate_%s'%j] = sku_14['sku_t4_counts_%s'%j] / (sku_14['sku_t6_counts_27_%s'%j]+1)
    sku_14['sku_14_t2_t1_rate_%s'%j] = sku_14['sku_t2_counts_%s'%j] / (sku_14['sku_t1_counts_%s'%j]+1)
    sku_14['sku_14_t2_t3_rate_%s'%j] = sku_14['sku_t2_counts_%s'%j] / (sku_14['sku_t3_counts_%s'%j]+1)
    sku_14['sku_14_t2_t5_rate_%s'%j] = sku_14['sku_t2_counts_%s'%j] / (sku_14['sku_t5_counts_%s'%j]+1)
    sku_14['sku_14_t2_t6_rate_%s'%j] = sku_14['sku_t2_counts_%s'%j] / (sku_14['sku_t6_counts_%s'%j]+1)
    
    sku_14.to_csv(jdata_path+'/output/%s_sku_%s_4.csv'%(nowtime,name[i]),index=False)
#    sku_5 = pd.read_csv(jdata_path+'/output/%s_sku_%s_0.csv'%(nowtime,name[i]))
#    sku_4 = pd.read_csv(jdata_path+'/output/%s_sku_%s_1.csv'%(nowtime,name[i]))
#    sku_3 = pd.read_csv(jdata_path+'/output/%s_sku_%s_2.csv'%(nowtime,name[i]))
#    sku_2 = pd.read_csv(jdata_path+'/output/%s_sku_%s_3.csv'%(nowtime,name[i]))
    sku_1 = pd.read_csv(jdata_path+'/output/%s_sku_%s_4.csv'%(nowtime,name[i]))
    
    sku_table = pd.merge(product,sku_1,on='sku_id',how='left')
#    sku_table = pd.merge(sku_table,sku_2,on='sku_id',how='left')
#    sku_table = pd.merge(sku_table,sku_3,on='sku_id',how='left')
#    sku_table = pd.merge(sku_table,sku_4,on='sku_id',how='left')
#    sku_table = pd.merge(sku_table,sku_5,on='sku_id',how='left')
    
#    sku_table['sku_t4_counts_0_1']=sku_table['sku_t4_counts_0']-sku_table['sku_t4_counts_1']
#    sku_table['sku_t4_counts_1_2']=sku_table['sku_t4_counts_1']-sku_table['sku_t4_counts_2']
#    sku_table['sku_t4_counts_2_3']=sku_table['sku_t4_counts_2']-sku_table['sku_t4_counts_3']
#    sku_table['sku_t4_counts_3_4']=sku_table['sku_t4_counts_3']-sku_table['sku_t4_counts_4']
#    sku_table['sku_t4_counts_0_1_rate'] = sku_table['sku_t4_counts_0_1']/sku_table['sku_t4_counts_0_1'].sum()
#    sku_table['sku_t4_counts_1_2_rate'] = sku_table['sku_t4_counts_1_2']/sku_table['sku_t4_counts_1_2'].sum()
#    sku_table['sku_t4_counts_2_3_rate'] = sku_table['sku_t4_counts_2_3']/sku_table['sku_t4_counts_2_3'].sum()
#    sku_table['sku_t4_counts_3_4_rate'] = sku_table['sku_t4_counts_3_4']/sku_table['sku_t4_counts_3_4'].sum()
    
    action_201604_cate8 = action_201604[action_201604.cate ==8]
    sku_table_user = action_201604_cate8.groupby('sku_id').user_id.unique().reset_index()
    sku_table_user['sku_user_count'] = sku_table_user.user_id.apply(lambda x:len(x))
    sku_table_user=sku_table_user[['sku_id','sku_user_count']]
    sku_table = pd.merge(sku_table,sku_table_user,how='left') 
    
    action_201604_1 = action_201604[action_201604.time >time[i*2]]
    action_201604_1['date'] = action_201604_1.time.apply(lambda x:x[:10])
    
    action_201604_dum = pd.get_dummies(action_201604_1.type,prefix='type')
    action_201604_dum = action_201604_1.join(action_201604_dum)
    action_201604_dum['user_id'] = action_201604_dum.user_id.astype(int)
    
    a_pd_hot = sku_table[sku_table['sku_t4_counts_0']>=10]
    a_pd_hot['hot']=1
    a_pd_nohot = sku_table[sku_table['sku_t4_counts_0']<10]
    a_pd_nohot['hot']=0
    sku_table= pd.concat([a_pd_hot,a_pd_nohot])    
    
    #日均行为数量
    for u in range(6):
        m=u+1
        ac1 = action_201604_dum.groupby(['sku_id','date']).sum()['type_%s'%m].unstack()
        ac1 =ac1.fillna(0)
        ac1_average=pd.DataFrame(ac1.mean(axis=1),columns=['sku_date_average_%s'%m]).reset_index()
        sku_table=pd.merge(sku_table,ac1_average,on='sku_id',how='left')
        
        ac2 = action_201604_dum.groupby(['sku_id','user_id']).sum()['type_%s'%m].unstack()
        ac2 =ac2.fillna(0)
        ac2_average=pd.DataFrame(ac2.mean(axis=1),columns=['sku_user_average_%s'%m]).reset_index()
        sku_table=pd.merge(sku_table,ac2_average,on='sku_id',how='left')
    
        a = action_201604_dum.groupby(['sku_id','date'],as_index=False)['type_%s'%m].sum()
        a.columns=['sku_id','date','type_%s_sum'%m]
        a = a.groupby('sku_id',as_index=False)['type_%s_sum'%m].mean()
        a.columns=['sku_id','type_%s_sku_date_average'%m]
        sku_table=pd.merge(sku_table,a,on='sku_id',how='left')
        
        a = action_201604_dum.groupby(['sku_id','user_id'],as_index=False)['type_%s'%m].sum()
        a.columns=['sku_id','user_id','type_%s_sum'%m]
        a0 = a.groupby('sku_id',as_index=False)['type_%s_sum'%m].mean()
        a0.columns=['sku_id','type_%s_sku_user_average'%m]
        sku_table=pd.merge(sku_table,a0,on='sku_id',how='left')

        a1 = a.groupby('sku_id',as_index=False)['type_%s_sum'%m].std()
        a1.columns=['sku_id','type_%s_sku_user_std'%m]
        sku_table=pd.merge(sku_table,a1,on='sku_id',how='left')  
        
        a2 = a.groupby('sku_id',as_index=False)['type_%s_sum'%m].median()
        a2.columns=['sku_id','type_%s_sku_user_median'%m]
        sku_table=pd.merge(sku_table,a2,on='sku_id',how='left')
        
    sku_table = sku_table.fillna(0)
    sku_table.to_csv(jdata_path+'/output/%s_sku_%s.csv'%(nowtime,name[i]),index=False)