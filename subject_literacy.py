# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:13:43 2017

@author: zhuangzijun
"""

import pandas as pd
import numpy as np

def main():
    df01 = pd.read_excel('./data/subject_quality/grade8-math201701.xls', skiprows=1)
    df05 = pd.read_excel('./data/subject_quality/grade8-math201705.xls', skiprows=1)
    df07 = pd.read_excel('./data/subject_quality/grade8-math201707.xls')
    
#    df01 = df01.iloc[1:, 41:69]
#    df01.columns = ['直观', '直观', '运算', '直观', '推理', '直观', '运算', '数据处理', '推理', '推理', '运算', '运算', '推理', '推理', '推理', '直观', '运算', '运算', '推理', '运算', '推理', '推理', '推理', '推理', '建模', '推理', '推理', '运算']
    # drop text
#    df01.drop(df01.index[-2], inplace=True)
#    df01 = df01[:-1].div(df01.iloc[-1, :])
    # .mean() only numeric type avaliable
#    df01 = df01.groupby(df01.columns, axis=1).apply(np.mean, axis=1)
#    df01 = data_process(df01, -2)
#    print(df01)
    
#    stu_df = df05.iloc[1:, 0:2]
#    stu_df.columns = ['教育ID', '姓名']
#    stu_df['教育ID'] = stu_df['教育ID'][:-2].astype(int).astype(str)
#    print(stu_df)
#    df05 = df05.iloc[1:, 39:65]
#    df05.columns = ['推理', '推理', '推理', '直观', '直观', '推理', '直观', '推理', '推理', '建模', '推理', '推理', '运算', '直观', '建模', '建模', '直观', '直观', '运算', '推理', '运算', '建模', '运算', '直观', '推理', '推理']
#    df05 = pd.merge(stu_df, df05, left_index=True, right_index=True)
#    df05.drop(df05.index[-1], inplace=True)
    
    df01 = data_process(df01, 41, 69, ['直观', '直观', '运算', '直观', '推理', '直观', '运算', '数据处理', '推理', '推理', '运算', '运算', '推理', '推理', '推理', '直观', '运算', '运算', '推理', '运算', '推理', '推理', '推理', '推理', '建模', '推理', '推理', '运算'], -2)
    df05 = data_process(df05, 39, 65, ['推理', '推理', '推理', '直观', '直观', '推理', '直观', '推理', '推理', '建模', '推理', '推理', '运算', '直观', '建模', '建模', '直观', '直观', '运算', '推理', '运算', '建模', '运算', '直观', '推理', '推理'], -1)
    print(df01)
#    df05 = data_process(df05, -1)
    
    #print(df05)
 
   
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
    
    return df
    
if __name__ == "__main__": main() 