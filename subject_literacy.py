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
    
    result05 = exams_process(df01, df05)
    result07 = exams_process(df05, df07)


def exams_process(df_last, df_now):
 #   print(df05.head())
 #   print(df07.head())
     
    # 两次成绩
    df = pd.merge(df_now, df_last, left_index=True, right_index=True, how='left')
    
    # 按权重算的201707成绩
    df[['建模weight', '推理weight', '数据处理weight', '直观weight', '运算weight']] = 2/3*df[['建模_x', '推理_x', '数据处理_x', '直观_x', '运算_x']] + 1/3*df[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    
    # 原始的成绩变化
    df[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] = df[['建模_x', '推理_x', '数据处理_x', '直观_x', '运算_x']] - df[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    # 权重的成绩变化
    df[['diff建模weight', 'diff推理weight', 'diff数据处理weight', 'diff直观weight', 'diff运算weight']] = df[['建模weight', '推理weight', '数据处理weight', '直观weight', '运算weight']] - df[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    
    # 避免被除数出现0的情况
    df.ix[df['建模_y'] == 0, '建模_y'] = 0.01
    df.ix[df['推理_y'] == 0, '推理_y'] = 0.01
    df.ix[df['数据处理_y'] == 0, '数据处理_y'] = 0.01
    df.ix[df['直观_y'] == 0, '直观_y'] = 0.01
    df.ix[df['运算_y'] == 0, '运算_y'] = 0.01
 
    df_diff = df[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] / df[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    df_diff_weight = df[['diff建模weight', 'diff推理weight', 'diff数据处理weight', 'diff直观weight', 'diff运算weight']] / df[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values

    orignal = bigger_than_20_percent(df_diff)
    new = bigger_than_20_percent(df_diff_weight)
    data = []
    data.append(orignal)
    data.append(new)
    result_df = pd.DataFrame(data, columns=['建模', '推理', '数据处理', '直观', '运算'], index=['原始成绩变化幅度>20%', '2:1权重成绩变化幅度>20%'])
    print(result_df)
    return result_df
        
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