# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:13:43 2017

@author: zhuangzijun
"""

import pandas as pd
import numpy as np

def main():
    df01 = pd.read_excel('./data/subject_literacy/grade8-math201701.xls', skiprows=1)
    df05 = pd.read_excel('./data/subject_literacy/grade8-math201705.xls', skiprows=1)
    df07 = pd.read_excel('./data/subject_literacy/grade8-math201707.xls', skiprows=1)
    
    df01 = data_process(df01, 41, 69, ['直观', '直观', '运算', '直观', '推理', '直观', '运算', '数据处理', '推理', '推理', '运算', '运算', '推理', '推理', '推理', '直观', '运算', '运算', '推理', '运算', '推理', '推理', '推理', '推理', '建模', '推理', '推理', '运算'], -2)
    df05 = data_process(df05, 39, 65, ['推理', '推理', '推理', '直观', '直观', '推理', '直观', '推理', '推理', '建模', '推理', '推理', '运算', '直观', '建模', '建模', '直观', '直观', '运算', '推理', '运算', '建模', '运算', '直观', '推理', '推理'], -1)
    df07 = data_process(df07, 48, 83, ['推理', '直观', '运算', '建模', '数据处理', '推理', '推理', '建模', '推理', '直观', '直观', '运算', '运算', '运算', '推理', '推理', '运算', '运算', '运算', '运算', '推理', '运算', '运算', '建模', '数据处理', '数据处理', '数据处理', '推理', '推理', '推理', '推理', '推理', '运算', '运算', '运算'], -1)
    
    # 没有考到的素养沿用上一次考试该素养的值
    df05 = pd.merge(df05, df01[['数据处理']], left_index=True, right_index=True, how='left')
    
    result05_df, df05_weight = exams_process(df01, df01, df05, 2/3, 'old:now=2:1')
 #   print(df05.head())
 #   print(df05_weight.head())
    print(result05_df)
 #   result05_df.columns = pd.MultiIndex.from_product([['test201705'], result05_df.columns])
    result07_df, df07_weight = exams_process(df05, df05_weight, df07, 2/3, 'old:now=2:1')
    print(result07_df)
 #   result07_df.columns = pd.MultiIndex.from_product([['test201707'], result07_df.columns])
 #   print(result05_df)
 #   print(result07_df)
 #   result_merged_df = pd.merge(result05_df, result07_df, left_index=True, right_index=True)
 #   print(result_merged_df)
#    result_merged_df.to_csv('./data/学科素养规则改进数据模拟.csv')

    
def exams_process(df_last, df_last_weight, df_now, weight, str_weight):
 #   print(df_last.head())
 #   print(df_now.head())
     
    # 两次成绩(unweight)
    df = pd.merge(df_now, df_last, left_index=True, right_index=True, how='left')
    # 原始的成绩相对于上次考试的成绩变化
    df[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] = df[['建模_x', '推理_x', '数据处理_x', '直观_x', '运算_x']] - df[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    # 避免被除数出现0的情况
    df_dividend = df[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']]
    df_dividend = avoid_dividend_be_0(df_dividend)
    # 成绩变化幅度
    df = df[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] / df_dividend[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    original = bigger_than_20_percent(df)   
    # 两次成绩(weight)
    df_weight = pd.merge(df_now, df_last_weight, left_index=True, right_index=True, how='left')
    # 本次按权重算出来的成绩
    df_weight[['建模weight', '推理weight', '数据处理weight', '直观weight', '运算weight']] = (1-weight)*df_weight[['建模_x', '推理_x', '数据处理_x', '直观_x', '运算_x']] + weight*df_weight[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    df_now_weight = df_weight[['建模weight', '推理weight', '数据处理weight', '直观weight', '运算weight']]
    df_now_weight.rename(columns={'建模weight': '建模', '推理weight': '推理', '直观weight': '直观', '运算weight': '运算', '数据处理weight': '数据处理'}, inplace=True)
    # 本次按权重算出来的成绩相对于上次考试的成绩变化
    df_weight[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] = df_weight[['建模weight', '推理weight', '数据处理weight', '直观weight', '运算weight']] - df_weight[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    # 避免被除数出现0的情况
    df_weight_dividend = df_weight[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']]
    df_weight_dividend = avoid_dividend_be_0(df_weight_dividend)
    # 成绩变化幅度
    df_weight = df_weight[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] / df_weight_dividend[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    new = bigger_than_20_percent(df_weight)
    data = []
    data.append(original)
    data.append(new)
    result_df = pd.DataFrame(data, columns=['建模', '推理', '数据处理', '直观', '运算'], index=['原始成绩变化幅度>20%', str_weight+'权重成绩变化幅度>20%'])
    
    return result_df, df_now_weight

def avoid_dividend_be_0(df):
    for i, colname in enumerate(df.columns.tolist()):
        df.ix[df[colname] == 0, colname] = 0.01
    
    return df
         
def bigger_than_20_percent(df):
    data = []
    for i, col_name in enumerate(df.columns.tolist()):
        data.append(df[df[col_name].abs() > 0.2].shape[0]/df.shape[0])
    
 #   print(data)
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