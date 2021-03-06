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

def physics():
    df01, df05, df07 = read_data('physics', 'xlsx')
    # 每个学生学科素养的平均得分率
    df01 = data_process_with_sheets(df01, 62, 111)
    df05 = data_process_with_sheets(df05, 62, 111)
    df07 = data_process_with_sheets(df07, 70, 127)
    return df01, df05, df07

def math():
    df01, df05, df07 = read_data('math', 'xls')
    # 每个学生学科素养的平均得分率
    df01 = data_process(df01, 41, 69, ['直观', '直观', '运算', '直观', '推理', '直观', '运算', '数据处理', '推理', '推理', '运算', '运算', '推理', '推理', '推理', '直观', '运算', '运算', '推理', '运算', '推理', '推理', '推理', '推理', '建模', '推理', '推理', '运算'], -2)
    df05 = data_process(df05, 39, 65, ['推理', '推理', '推理', '直观', '直观', '推理', '直观', '推理', '推理', '建模', '推理', '推理', '运算', '直观', '建模', '建模', '直观', '直观', '运算', '推理', '运算', '建模', '运算', '直观', '推理', '推理'], -1)
    df07 = data_process(df07, 48, 83, ['推理', '直观', '运算', '建模', '数据处理', '推理', '推理', '建模', '推理', '直观', '直观', '运算', '运算', '运算', '推理', '推理', '运算', '运算', '运算', '运算', '推理', '运算', '运算', '建模', '数据处理', '数据处理', '数据处理', '推理', '推理', '推理', '推理', '推理', '运算', '运算', '运算'], -1)
    return df01, df05, df07

def read_data(subject, suffix):
    df01 = pd.read_excel('./data/subject_literacy/grade8-'+subject+'201701.'+suffix, skiprows=1)
    df05 = pd.read_excel('./data/subject_literacy/grade8-'+subject+'201705.'+suffix, skiprows=1)
    df07 = pd.read_excel('./data/subject_literacy/grade8-'+subject+'201707.'+suffix, skiprows=1)
    return df01, df05, df07
    
def main():
    # df01, df05, df07 = physics()
    df01, df05, df07 = math()
    
    suyang05_weight_2to1 = cal_suyang(df01, df05, 2, 1)
    suyang05_weight_3to1 = cal_suyang(df01, df05, 3, 1)
    suyang05_weight_4to1 = cal_suyang(df01, df05, 4, 1)
    suyang07_weight_2to1 = cal_suyang(suyang05_weight_2to1, df07, 2, 1)
    suyang07_weight_3to1 = cal_suyang(suyang05_weight_3to1, df07, 3, 1)
    suyang07_weight_4to1 = cal_suyang(suyang05_weight_4to1, df07, 4, 1)
    
    rangeablity_01to05 = []
    rangeablity_01to05.append(rangeablity(df01, df05))
    rangeablity_01to05.append(rangeablity(df01, suyang05_weight_2to1))
    rangeablity_01to05.append(rangeablity(df01, suyang05_weight_3to1))
    rangeablity_01to05.append(rangeablity(df01, suyang05_weight_4to1))
    rangeablity_01to05_df = pd.DataFrame(rangeablity_01to05, columns=suyang05_weight_2to1.columns.tolist(), index=['原始成绩变化幅度>20%', '2:1权重成绩变化幅度>20%', '3:1权重成绩变化幅度>20%', '4:1权重成绩变化幅度>20%'])
    print(rangeablity_01to05_df)
    
    rangeablity_05to07 = []
    rangeablity_05to07.append(rangeablity(df05, df07))
    rangeablity_05to07.append(rangeablity(suyang05_weight_2to1, suyang07_weight_2to1))
    rangeablity_05to07.append(rangeablity(suyang05_weight_3to1, suyang07_weight_3to1))
    rangeablity_05to07.append(rangeablity(suyang05_weight_4to1, suyang07_weight_4to1))
    rangeablity_01to07_df = pd.DataFrame(rangeablity_05to07, columns=suyang07_weight_2to1.columns.tolist(), index=['原始成绩变化幅度>20%', '2:1权重成绩变化幅度>20%', '3:1权重成绩变化幅度>20%', '4:1权重成绩变化幅度>20%'])
    print(rangeablity_01to07_df)
    
    subject_literacy_line_chart(df01, df05, df07, df01, suyang05_weight_2to1, suyang07_weight_2to1)
    
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
    
def cal_suyang(df_last, df_this, w1, w2):
    last_cols = df_last.columns.tolist()
    this_cols = df_this.columns.tolist()
    # 两次考试素养的集合
    total_cols = list(set(last_cols + this_cols))
    df_this.columns = list(map(lambda x: x+'_x', df_this.columns.tolist()))
    df_last.columns = list(map(lambda y: y+'_y', df_last.columns.tolist()))
    df = pd.merge(df_this, df_last, left_index=True, right_index=True)
    for i, col in enumerate(total_cols):
        if (col in last_cols) & (col not in this_cols):
            df[col] = df[col+'_y'] 
            df_this[col+'_y'] = df[col]
        elif (col not in last_cols) & (col in this_cols):
            df[col] = df[col+'_x'] 
        elif (col in last_cols) & (col in this_cols):
            df[col] = w1/(w1+w2) * df[col+'_y'] + w2/(w1+w2) * df[col+'_x']
    
    df_this.columns = list(map(lambda x: x[:-2], df_this.columns.tolist()))
    df_last.columns = list(map(lambda y: y[:-2], df_last.columns.tolist()))        
    df = df[total_cols]
    return df

def rangeablity(df_last, df_this):
    # 两次成绩(unweight)
    df = pd.merge(df_this, df_last, left_index=True, right_index=True)
    columns = list(set(df_last.columns.tolist() + df_this.columns.tolist()))
    columns_diff = list(map(lambda x: x+'_diff', columns))
    columns_x = list(map(lambda x: x+'_x', columns))
    columns_y = list(map(lambda x: x+'_y', columns))
    # 原始的成绩相对于上次考试的成绩变化
    df[columns_diff] = df[columns_x] - df[columns_y].values
    # print(df[columns_diff])
    # 避免被除数出现0的情况
    df_dividend = df[columns_y]
    df_dividend = avoid_dividend_be_0(df_dividend)
    # 成绩变化幅度
    df = df[columns_diff] / df_dividend[columns_y].values
    rangeability = bigger_than_20_percent(df)
    
    return rangeability
    
def subject_literacy_line_chart(df01, df05, df07, df01weight_2to1, df05weight_2to1, df07weight_2to1):
    origin_df, weight_2to1_df = subject_literacy(df01, df05, df07, df01weight_2to1, df05weight_2to1, df07weight_2to1)
    plot_line_chart(origin_df, '原始学科素养计算')
    plot_line_chart(weight_2to1_df, '2：1权重学科素养计算')

def plot_line_chart(df, title):
    ax = df.plot(kind='line')
    ax.set_ylim(0,1)
    ax.set_title(title, fontproperties=chinesefont)
    plt.legend(prop=chinesefont)
    plt.show()
    
def subject_literacy(df01, df05, df07, df01weight_2to1, df05weight_2to1, df07weight_2to1):
    origin_df = avg_subject_literacy(df01, df05, df07) 
    weight_2to1_df = avg_subject_literacy(df01weight_2to1, df05weight_2to1, df07weight_2to1)
    return origin_df, weight_2to1_df
     
def avg_subject_literacy(df_first, df_second, df_third):
    columns = list(set(df_first.columns.tolist() + df_second.columns.tolist() + df_third.columns.tolist()))
    data = []
    data.append(df_col_mean(df_first))
    data.append(df_col_mean(df_second))
    data.append(df_col_mean(df_third))
    df = pd.DataFrame(data, columns=columns)
    df.set_index([['201701', '201705', '201707']], inplace=True)
    return df
    
def df_col_mean(df):
    data = {}
    for i, colname in enumerate(df.columns.tolist()):
        data[colname] = df[colname].mean()
    
    return data

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