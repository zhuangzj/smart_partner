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
    # rename columns' names
    df01.rename(columns={'建模': '201701建模', '推理': '201701推理', '直观': '201701直观', '运算': '201701运算', '数据处理': '201701数据处理'}, inplace=True)
    df05.rename(columns={'建模': '201705建模', '推理': '201705推理', '直观': '201705直观', '运算': '201705运算', '数据处理': '201705数据处理'}, inplace=True)    
    
    # 两次成绩
    df = pd.merge(df05, df01, left_index=True, right_index=True, how='left')
    
    # 按权重算的201705成绩
    df[['201705建模weight', '201705推理weight', '201705数据处理weight', '201705直观weight', '201705运算weight']] = 2/3*df[['201705建模', '201705推理', '201705数据处理', '201705直观', '201705运算']] + 1/3*df[['201701建模', '201701推理', '201701数据处理', '201701直观', '201701运算']].values
    
    # 原始201705和201701的成绩变化
    df[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] = df[['201705建模', '201705推理', '201705数据处理', '201705直观', '201705运算']] - df[['201701建模', '201701推理', '201701数据处理', '201701直观', '201701运算']].values
    
    # 权重201705和201701的成绩变化
    df[['diff建模weight', 'diff推理weight', 'diff数据处理weight', 'diff直观weight', 'diff运算weight']] = df[['201705建模weight', '201705推理weight', '201705数据处理weight', '201705直观weight', '201705运算weight']] - df[['201701建模', '201701推理', '201701数据处理', '201701直观', '201701运算']].values
    
    df.ix[df['201701建模'] == 0, '201701建模'] = 0.01
    df.ix[df['201701推理'] == 0, '201701推理'] = 0.01
    df.ix[df['201701直观'] == 0, '201701直观'] = 0.01
    df.ix[df['201701运算'] == 0, '201701运算'] = 0.01
    df.ix[df['201701数据处理'] == 0, '201701数据处理'] = 0.01
    print(df)
    df_diff = df[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] / df[['201701建模', '201701推理', '201701数据处理', '201701直观', '201701运算']].values
    df_diff_weight = df[['diff建模weight', 'diff推理weight', 'diff数据处理weight', 'diff直观weight', 'diff运算weight']] / df[['201701建模', '201701推理', '201701数据处理', '201701直观', '201701运算']].values
    #print(df)
    original = []
    original.append(df_diff[df_diff['diff建模'].abs() > 0.2].shape[0]/df.shape[0])
    original.append(df_diff[df_diff['diff推理'].abs() > 0.2].shape[0]/df.shape[0])
    original.append(df_diff[df_diff['diff数据处理'].abs() > 0.2].shape[0]/df.shape[0])
    original.append(df_diff[df_diff['diff直观'].abs() > 0.2].shape[0]/df.shape[0])
    original.append(df_diff[df_diff['diff运算'].abs() > 0.2].shape[0]/df.shape[0])
    print(original)
    new = []
    new.append(df_diff_weight[df_diff_weight['diff建模weight'].abs() > 0.2].shape[0]/df.shape[0])
    new.append(df_diff_weight[df_diff_weight['diff推理weight'].abs() > 0.2].shape[0]/df.shape[0])
    new.append(df_diff_weight[df_diff_weight['diff数据处理weight'].abs() > 0.2].shape[0]/df.shape[0])
    new.append(df_diff_weight[df_diff_weight['diff直观weight'].abs() > 0.2].shape[0]/df.shape[0])
    new.append(df_diff_weight[df_diff_weight['diff运算weight'].abs() > 0.2].shape[0]/df.shape[0])
    print(new)
 #  df05_merged[['diff_建模','diff_推理','diff_直观','diff_运算','diff_数据处理']] = (df05_merged[['建模_x','推理_x','直观_x','运算_x','数据处理_x']] - df05_merged[['建模_y','推理_y','直观_y','运算_y','数据处理_y']].values)/df05_merged[['建模_y','推理_y','直观_y','运算_y','数据处理_y']].values
 #   print(df05_merged)
 #   original_unstable_number = df05_merged[df05_merged[['diff_建模','diff_推理','diff_直观','diff_运算','diff_数据处理']].abs() > 0.2].dropna(axis=0, how='all').shape[0]
 #   original_percentage = original_unstable_number / df05_merged.shape[0]
 #   print(original_percentage)
 #   df05_merged[['建模_byweight','推理_byweight','直观_byweight','运算_byweight','数据处理_byweight']] = 2/3*df05_merged[['建模_x','推理_x','直观_x','运算_x','数据处理_x']] + 1/3*df05_merged[['建模_y','推理_y','直观_y','运算_y','数据处理_y']].values
 #   df05_merged[['diff_建模','diff_推理','diff_直观','diff_运算','diff_数据处理']] = (df05_merged[['建模_byweight','推理_byweight','直观_byweight','运算_byweight','数据处理_byweight']] - df05_merged[['建模_y','推理_y','直观_y','运算_y','数据处理_y']].values)/df05_merged[['建模_y','推理_y','直观_y','运算_y','数据处理_y']].values
 #   updated_unstable_number = df05_merged[df05_merged[['diff_建模','diff_推理','diff_直观','diff_运算','diff_数据处理']].abs() > 0.2].dropna(axis=0, how='all').shape[0]
 #   updated_percentage = updated_unstable_number / df05_merged.shape[0]
 #   print(updated_percentage)
 #   print(df05_merged)
    # 将201701的得分率为0的转成0.01
    #df_rangeablity = (df['test201705'][['建模', '推理', '数据处理', '直观', '运算']] - df['test201701'][['建模', '推理', '数据处理', '直观', '运算']].values)/df['test201701'][['建模', '推理', '数据处理', '直观', '运算']].values
    #print(df_merged)
    #df['diff'] = df['test201705'] - df['test201701']


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