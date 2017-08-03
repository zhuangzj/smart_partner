# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 08:56:03 2017

@author: zhuangzijun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

chinesefont = font_manager.FontProperties()
chinesefont.set_family('SimHei')

def main():
    df01 = pd.read_excel('./data/subject_literacy/grade8-physics201701.xlsx', skiprows=1)
    df05 = pd.read_excel('./data/subject_literacy/grade8-physics201705.xlsx', skiprows=1)
    df07 = pd.read_excel('./data/subject_literacy/grade8-physics201707.xlsx', skiprows=1)
    
    # 每个学生学科素养的平均得分率
    df01 = data_process_with_sheets(df01, 62, 111)
    df05 = data_process_with_sheets(df05, 62, 111)
    df07 = data_process_with_sheets(df07, 70, 127)    

    suyang05_weight_2to1 = cal_suyang_weight(df01, df05, 2, 1)
    suyang07_weight_2to1 = cal_suyang_weight(df05, df07, 2, 1)
    suyang05_weight_3to1 = cal_suyang_weight(df01, df05, 3, 1)
    suyang07_weight_3to1 = cal_suyang_weight(df05, df07, 3, 1)
    
    subject_literacy_line_chart(df01, df05, df07, suyang05_weight_2to1, suyang05_weight_3to1, suyang07_weight_2to1, suyang07_weight_3to1)
    
def data_process_with_sheets(df, start, end):
    temp_df = df
    columns = df.iloc[0, start:end].tolist()
    df = df.iloc[2:, start:end]
    df.columns = columns
    
    stu_df = temp_df.iloc[2:, 0:2]
    stu_df.columns = ['教育ID', '姓名']
    df = pd.merge(stu_df, df, left_index=True, right_index=True)
    
    df.set_index(['教育ID', '姓名'], inplace=True)
    
    # score rate
    df = df[:-1].div(df.iloc[-1, :])
    # avg score rate of each subject literacy
    df = df.groupby(df.columns, axis=1).apply(np.mean, axis=1)
    
    return df
    
def cal_suyang_weight(df_last, df_this, w1, w2):
    last_cols = df_last.columns
    this_cols = df_this.columns
    # 两次考试素养的集合
    total_cols = last_cols + this_cols
    df = pd.merge(df_this, df_last, left_index=True, right_index=True)
    for i, col in enumerate(total_cols):
        if (col in last_cols) & (col not in this_cols):
            df[col] = df[col+'_y'] 
        elif (col not in last_cols) & (col in this_cols):
            df[col] = df[col+'_x'] 
        elif (col in last_cols) & (col in this_cols):
            df[col] = w1/(w1+w2) * df[col+'_y'] + w2/(w1+w2) * df[col+'_x']
    df = df[total_cols]
    return df, total_cols

def rangeablity_df(df_last, df_this, columns):
    # 两次成绩(unweight)
    df = pd.merge(df_this, df_last, left_index=True, right_index=True)
    # 原始的成绩相对于上次考试的成绩变化
    columns_diff = list(map(lambda x: x+'_diff', columns))
    columns_x = list(map(lambda x: x+'_x', columns))
    columns_y = list(map(lambda x: x+'_y', columns))
    df[columns_diff] = df[['实验探究_x', '物理观念_x']] - df[['实验探究_y', '物理观念_y']].values
    # 避免被除数出现0的情况
    df_dividend = df[['实验探究_y', '物理观念_y']]
    df_dividend = avoid_dividend_be_0(df_dividend)
    # 成绩变化幅度
    df = df[['diff实验探究', 'diff物理观念']] / df_dividend[['实验探究_y', '物理观念_y']].values
    rangeability = bigger_than_20_percent(df)
    
    return rangeability
    
def subject_literacy_line_chart(df01, df05, df07, df05weight_2to1, df05weight_3to1, df07weight_2to1, df07weight_3to1):
    origin_df, weight_2to1_df = subject_literacy(df01, df05, df07, df05weight_2to1, df05weight_3to1, df07weight_2to1, df07weight_3to1)
    plot_line_chart(origin_df, '原始学科素养计算')
    plot_line_chart(weight_2to1_df, '2：1权重学科素养计算')

def plot_line_chart(df, title):
    ax = df.plot(kind='line')
    ax.set_ylim(0,1)
    ax.set_title(title, fontproperties=chinesefont)
    plt.legend(prop=chinesefont)
    plt.show()
    
def subject_literacy(df01, df05, df07, df05weight_2to1, df05weight_3to1, df07weight_2to1, df07weight_3to1):
#    print('origin')
    origin_df = avg_subject_literacy(df01, df05, df07) 
#    print('weigh2:1')
    weight_2to1_df = avg_subject_literacy(df01, df05weight_2to1, df07weight_2to1)
#    print('weigh3:1')
#    weight_3to1_subject_literacy(df01, df05weight_3to1, df07weight_3to1)
    return origin_df, weight_2to1_df
     
def avg_subject_literacy(df01, df05, df07):
    data = []
    data.append(df_col_mean(df01))
    data.append(df_col_mean(df05))
    data.append(df_col_mean(df07))
    df = pd.DataFrame(data, columns=('实验探究', '物理观念'))
    df.set_index([['201701', '201705', '201707']], inplace=True)
 #   print(df)
    return df
    
def df_col_mean(df):
    data = {}
    for i, colname in enumerate(df.columns.tolist()):
        #print(colname)
        data[colname] = df[colname].mean()
    
    return data

def rangeablity_df(df_last, df_last_weight_2to1, df_last_weight_3to1, df_last_weight_4to1, df):
    data = []
    # 两次原始数据计算成绩变化程度
    rangeability_raw = raw_rangeability(df_last, df)
    data.append(rangeability_raw)
    # 利用权重(2:1)计算成绩和成绩变化程度
    test_weight_2to1, rangeability_weight_2to1 = weight_rangeability(df_last_weight_2to1, df, 2/3)
    data.append(rangeability_weight_2to1)
    # 利用权重(3:1)计算成绩和成绩变化程度
    test_weight_3to1, rangeability_weight_3to1 = weight_rangeability(df_last_weight_3to1, df, 3/4)
    data.append(rangeability_weight_3to1)
    # 利用权重(4:1)计算成绩和成绩变化程度
    test_weight_4to1, rangeability_weight_4to1 = weight_rangeability(df_last_weight_4to1, df, 4/5)
    data.append(rangeability_weight_4to1)
    rangeability_df = pd.DataFrame(data, columns=['实验探究', '物理观念'], index=['原始成绩变化幅度>20%', '2:1权重成绩变化幅度>20%', '3:1权重成绩变化幅度>20%', '4:1权重成绩变化幅度>20%'])

    return rangeability_df, test_weight_2to1, test_weight_3to1, test_weight_4to1

def raw_rangeability(df_last, df_now):
    # 两次成绩(unweight)
    df = pd.merge(df_now, df_last, left_index=True, right_index=True, how='left')
    # 原始的成绩相对于上次考试的成绩变化
    df[['diff实验探究', 'diff物理观念']] = df[['实验探究_x', '物理观念_x']] - df[['实验探究_y', '物理观念_y']].values
    # 避免被除数出现0的情况
    df_dividend = df[['实验探究_y', '物理观念_y']]
    df_dividend = avoid_dividend_be_0(df_dividend)
    # 成绩变化幅度
    df = df[['diff实验探究', 'diff物理观念']] / df_dividend[['实验探究_y', '物理观念_y']].values
    rangeability = bigger_than_20_percent(df)
    
    return rangeability

def weight_rangeability(df_last_weight, df_now, weight):
   
    # 两次成绩(weight)
    df_weight = pd.merge(df_now, df_last_weight, left_index=True, right_index=True, how='left')
    # 本次按权重算出来的成绩
    df_weight[['实验探究weight', '物理观念weight']] = (1-weight)*df_weight[['实验探究_x', '物理观念_x']] + weight*df_weight[['实验探究_y', '物理观念_y']].values
    df_weight_score = df_weight[['实验探究weight', '物理观念weight']]
    df_weight_score.rename(columns={'实验探究weight': '实验探究', '物理观念weight': '物理观念'}, inplace=True)
    # 本次按权重算出来的成绩相对于上次考试的成绩变化
    df_weight[['diff实验探究', 'diff物理观念']] = df_weight[['实验探究weight', '物理观念weight']] - df_weight[['实验探究_y', '物理观念_y']].values
    # 避免被除数出现0的情况
    df_weight_dividend = df_weight[['实验探究_y', '物理观念_y']]
    df_weight_dividend = avoid_dividend_be_0(df_weight_dividend)
    # 成绩变化幅度
    df_weight = df_weight[['diff实验探究', 'diff物理观念']] / df_weight_dividend[['实验探究_y', '物理观念_y']].values
    rangeability = bigger_than_20_percent(df_weight)
    
    return df_weight_score, rangeability

def avoid_dividend_be_0(df):
    for i, colname in enumerate(df.columns.tolist()):
        df.ix[df[colname] == 0, colname] = 0.01
    
    return df
         
def bigger_than_20_percent(df):
    data = []
    for i, col_name in enumerate(df.columns.tolist()):
        data.append(df[df[col_name].abs() > 0.2].shape[0]/df.shape[0])
    
    return data

def data_process(df, start, end, columns, drop_row):
    temp_df = df

    df = df.iloc[1:, start:end]
    df.columns = columns
    # drop text
    df.drop(df.index[drop_row], inplace=True)
    # score rate
    df = df[:-1].div(df.iloc[-1, :])
    # avg score rate of each subject literacy
    df = df.groupby(df.columns, axis=1).apply(np.mean, axis=1) # .mean() only numeric type avaliable
    
    # get attributes of stus
    stu_df = temp_df.iloc[1:, 0:2]
    stu_df.columns = ['教育ID', '姓名']
    stu_df['教育ID'] = stu_df['教育ID'][:-2].astype(int).astype(str)
    df = pd.merge(stu_df, df, left_index=True, right_index=True)
    
    df.set_index(['教育ID', '姓名'], inplace=True)
    return df

if __name__ == "__main__": main() 