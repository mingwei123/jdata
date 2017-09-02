# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:26:33 2017

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
comment = pd.read_csv(jdata_path+"/data/JData_Comment.csv")

train_start_time = '2016-04-01'
train_end_time = '2016-04-06'
validate_start_time =train_end_time
validate_end_time = '2016-04-11'
test_start_time = validate_end_time
test_end_time = '2016-04-16'

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

#comment_badcomment = pd.get_dummies(comment.has_bad_comment,prefix='has_bad_comment')
#comment = comment.join(comment_badcomment)

nowtime =datetime.now().strftime("%Y%m%d")

for i in range(3):
    comment_time = comment[(comment.dt < str(datetime.strptime(time[i*2+1],'%Y-%m-%d')))&(comment.dt >time[i*2])]
    comment_sort = comment_time.sort(columns='dt',ascending=False)
    comment_sort_drop_dup = comment_sort.drop_duplicates('sku_id')
    del comment_sort_drop_dup['dt']
    com_0 = comment_sort_drop_dup[comment_sort_drop_dup.comment_num==0]
    com_1 = comment_sort_drop_dup[comment_sort_drop_dup.comment_num==1]
    com_2 = comment_sort_drop_dup[comment_sort_drop_dup.comment_num==2]
    com_2['comment_num'] = com_2['comment_num']*5
    com_3 = comment_sort_drop_dup[comment_sort_drop_dup.comment_num==3]
    com_3['comment_num'] = com_3['comment_num']*25
    com_4 = comment_sort_drop_dup[comment_sort_drop_dup.comment_num==4]
    com_4['comment_num'] = com_4['comment_num']*70
    comment_sort_drop_dup=pd.concat([com_0,com_1,com_2,com_3,com_4])
    comment_sort_drop_dup['bad_num']=comment_sort_drop_dup['comment_num']*comment_sort_drop_dup['bad_comment_rate']

    comment_sort_drop_dup.to_csv(jdata_path+'/output/%s_comment_%s.csv'%(nowtime,name[i]),index=False)
     
