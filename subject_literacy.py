# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:13:43 2017

@author: zhuangzijun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

chinesefont = font_manager.FontProperties()
chinesefont.set_family('SimHei')

def main():
    df01 = pd.read_excel('./data/subject_literacy/grade8-math201701.xls', skiprows=1)
    df05 = pd.read_excel('./data/subject_literacy/grade8-math201705.xls', skiprows=1)
    df07 = pd.read_excel('./data/subject_literacy/grade8-math201707.xls', skiprows=1)
    
    df01 = data_process(df01, 41, 69, ['直观', '直观', '运算', '直观', '推理', '直观', '运算', '数据处理', '推理', '推理', '运算', '运算', '推理', '推理', '推理', '直观', '运算', '运算', '推理', '运算', '推理', '推理', '推理', '推理', '建模', '推理', '推理', '运算'], -2)
    df05 = data_process(df05, 39, 65, ['推理', '推理', '推理', '直观', '直观', '推理', '直观', '推理', '推理', '建模', '推理', '推理', '运算', '直观', '建模', '建模', '直观', '直观', '运算', '推理', '运算', '建模', '运算', '直观', '推理', '推理'], -1)
    df07 = data_process(df07, 48, 83, ['推理', '直观', '运算', '建模', '数据处理', '推理', '推理', '建模', '推理', '直观', '直观', '运算', '运算', '运算', '推理', '推理', '运算', '运算', '运算', '运算', '推理', '运算', '运算', '建模', '数据处理', '数据处理', '数据处理', '推理', '推理', '推理', '推理', '推理', '运算', '运算', '运算'], -1)
    
    # 没有考到的素养沿用上一次考试该素养的值
    df05 = pd.merge(df05, df01[['数据处理']], left_index=True, right_index=True, how='left')
    rangeability05_df, df05weight_2to1, df05weight_3to1, df05weight_4to1 = rangeablity_df(df01, df01, df01, df01, df05)
#    rangeability05_df.columns = pd.MultiIndex.from_product([['test201705'], rangeability05_df.columns])
    
    rangeability07_df, df07weight_2to1, df07weight_3to1, df07weight_4to1 = rangeablity_df(df05, df05weight_2to1, df05weight_3to1, df05weight_4to1, df07)
#    rangeability07_df.columns = pd.MultiIndex.from_product([['test201707'], rangeability07_df.columns])
#    rangeability_merged_df = pd.merge(rangeability05_df, rangeability07_df, left_index=True, right_index=True)
#    print(rangeability_merged_df)
#    rangeability_merged_df.to_csv('./data/学科素养规则改进数据模拟.csv')
     # 检验
#    valid_modeling(df01, df05, df07)
    # 学科素养曲线图
    subject_literacy_line_chart(df01, df05, df07, df05weight_2to1, df05weight_3to1, df07weight_2to1, df07weight_3to1)

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
     
def weight_3to1_subject_literacy(df01, df05weight_3to1, df07weight_3to1):
    df01_modeling = df01['建模'].mean()
    df05_modeling = df05weight_3to1['建模'].mean()
    df07_modeling = df07weight_3to1['建模'].mean()
    print(df01_modeling)
    print(df05_modeling)
    print(df07_modeling)

def avg_subject_literacy(df01, df05, df07):
    data = []
    data.append(df_col_mean(df01))
    data.append(df_col_mean(df05))
    data.append(df_col_mean(df07))
    df = pd.DataFrame(data, columns=('建模', '推理', '数据处理', '直观', '运算'))
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
    rangeability_df = pd.DataFrame(data, columns=['建模', '推理', '数据处理', '直观', '运算'], index=['原始成绩变化幅度>20%', '2:1权重成绩变化幅度>20%', '3:1权重成绩变化幅度>20%', '4:1权重成绩变化幅度>20%'])

    return rangeability_df, test_weight_2to1, test_weight_3to1, test_weight_4to1

def raw_rangeability(df_last, df_now):
    # 两次成绩(unweight)
    df = pd.merge(df_now, df_last, left_index=True, right_index=True, how='left')
    # 原始的成绩相对于上次考试的成绩变化
    df[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] = df[['建模_x', '推理_x', '数据处理_x', '直观_x', '运算_x']] - df[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    # 避免被除数出现0的情况
    df_dividend = df[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']]
    df_dividend = avoid_dividend_be_0(df_dividend)
    # 成绩变化幅度
    df = df[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] / df_dividend[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    rangeability = bigger_than_20_percent(df)
    
    return rangeability

def weight_rangeability(df_last_weight, df_now, weight):
   
    # 两次成绩(weight)
    df_weight = pd.merge(df_now, df_last_weight, left_index=True, right_index=True, how='left')
    # 本次按权重算出来的成绩
    df_weight[['建模weight', '推理weight', '数据处理weight', '直观weight', '运算weight']] = (1-weight)*df_weight[['建模_x', '推理_x', '数据处理_x', '直观_x', '运算_x']] + weight*df_weight[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    df_weight_score = df_weight[['建模weight', '推理weight', '数据处理weight', '直观weight', '运算weight']]
    df_weight_score.rename(columns={'建模weight': '建模', '推理weight': '推理', '直观weight': '直观', '运算weight': '运算', '数据处理weight': '数据处理'}, inplace=True)
    # 本次按权重算出来的成绩相对于上次考试的成绩变化
    df_weight[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] = df_weight[['建模weight', '推理weight', '数据处理weight', '直观weight', '运算weight']] - df_weight[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
    # 避免被除数出现0的情况
    df_weight_dividend = df_weight[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']]
    df_weight_dividend = avoid_dividend_be_0(df_weight_dividend)
    # 成绩变化幅度
    df_weight = df_weight[['diff建模', 'diff推理', 'diff数据处理', 'diff直观', 'diff运算']] / df_weight_dividend[['建模_y', '推理_y', '数据处理_y', '直观_y', '运算_y']].values
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

def valid_modeling(df01, df05, df07):
    df = pd.merge(df05, df01, left_index=True, right_index=True)
    print(df01.loc['13226593', '吕伟']['建模'])
    print(df05.loc['13226593', '吕伟']['建模'])
    print(df07.loc['13226593', '吕伟']['建模'])
#    print(df.loc['13226593', '吕伟'])
    data = []
    df = df[['建模_x', '建模_y']]
    df['建模_y_dividend'] = df['建模_y']
    df.ix[df['建模_y_dividend'] == 0, '建模_y_dividend'] = 0.01
    df['05建模rangeability'] = (df['建模_x'] - df['建模_y']) / df['建模_y_dividend']
    count = df[df['05建模rangeability'].abs() > 0.2].shape[0]
    percentage = count / df.shape[0]
    print(percentage)
    data.append(percentage)
    df['2:1建模_x'] = 1/3*df['建模_x'] + 2/3*df['建模_y']
    df['(2:1)05建模rangeability'] = (df['2:1建模_x'] - df['建模_y']) / df['建模_y_dividend']
    #print(df)
    count = df[df['(2:1)05建模rangeability'].abs() > 0.2].shape[0]
    percentage = count / df.shape[0]
    print(percentage)
    data.append(percentage)
    df['3:1建模_x'] = 1/4*df['建模_x'] + 3/4*df['建模_y']
    df['(3:1)05建模rangeability'] = (df['3:1建模_x'] - df['建模_y']) / df['建模_y_dividend']
    
    percentage = count / df.shape[0]
    print(percentage)
    data.append(percentage)
    df = pd.merge(df, df07[['建模', '运算']], left_index=True, right_index=True)
    df = df.iloc[:, 0:9]
    df['建模_x_dividend'] = df['建模_x']
    df.ix[df['建模_x_dividend'] == 0, '建模_x_dividend'] = 0.01
    df['07建模rangeability'] = (df['建模'] - df['建模_x']) / df['建模_x_dividend']
    #print(df)
    count = df[df['07建模rangeability'].abs() > 0.2].shape[0]
    percentage = count / df.shape[0]
    print(percentage)
    data.append(percentage)
    df['2:1建模_z'] = 1/4*df['建模'] + 3/4*df['2:1建模_x']
    df['2:1建模_x_dividend'] = df['2:1建模_x']
    df.ix[df['2:1建模_x_dividend'] == 0, '2:1建模_x_dividend'] = 0.01
    df['(2:1)07建模rangeability'] = (df['2:1建模_z'] - df['2:1建模_x']) / df['2:1建模_x_dividend']
    count = df[df['(2:1)07建模rangeability'].abs() > 0.2].shape[0]
    percentage = count / df.shape[0]
    print(percentage)
    data.append(percentage)
    df['3:1建模_z'] = 1/4*df['建模'] + 3/4*df['3:1建模_x']
    df['3:1建模_x_dividend'] = df['3:1建模_x']
    df.ix[df['3:1建模_x_dividend'] == 0, '3:1建模_x_dividend'] = 0.01
    df['(3:1)07建模rangeability'] = (df['3:1建模_z'] - df['3:1建模_x']) / df['3:1建模_x_dividend']
    df_bigthan3perc = df[df['(3:1)05建模rangeability'].abs() > 0.2]
    print(df_bigthan3perc[['建模_x', '建模_y', '建模']])
    print(df_bigthan3perc.loc['13226593', '吕伟'])
    print(df.loc['13226754', '赵云喆'])
    count = df[df['(3:1)07建模rangeability'].abs() > 0.2].shape[0]
    percentage = count / df.shape[0]
    print(percentage)
    data.append(percentage)
    
    
    
if __name__ == "__main__": main() 