# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:22:21 2017

@author: zhuangzijun
"""

import pandas as pd

def difficulty(df):
    
    df.loc['试卷得分']['总分'] = 100
    df.dropna(inplace=True)
    df['难度'] = 1 - df['平均分'] / df['总分']
    #print(df)
    return df

def get_q_total_score():
    test_notation = pd.read_excel('./data/tongzhou_201701/test_notation.xlsx', skiprows=2,
                              names=['题目编码', '得分点编码', '一级主题代码', '一级主题', '二级主题代码', '二级主题', '核心概念代码', '核心概念', '学习表现指标代码', '学习表现指标', '核心素养代码', '核心素养',
                                     '内容属性代码', '内容属性', '熟悉度代码', '熟悉度', '间接度代码', '间接度', '评分标准', '总分'],
                                     converters={'题目编码':str,'得分点编码':str})
    #print(test_notation)
    test_notation['题目编号'] = 'P' + '0' + test_notation['题目编码'].map(str) + '0' + test_notation['得分点编码'].map(str)
    return test_notation[['题目编号', '总分']]

def main():
    math_df = pd.read_excel('./data/tongzhou_201701/math.xlsx')
    math_df['试卷得分'] = math_df.iloc[:, 5:32].sum(axis=1)
    math_df = math_df.iloc[:, 5:34]
    #print(math_df.columns)
    
    # q-mean
    mean = math_df.mean()
    q_mean = pd.DataFrame({'题目编号':mean.index, '平均分':mean.values})
    #print(q_mean)
    
    # q-totalscore
    q_score = get_q_total_score()
    #print(q_score)
    
    df = q_mean.merge(q_score, how='left', on='题目编号')
    df.set_index(['题目编号'], inplace=True)
    
    df = difficulty(df)


    
if __name__ == "__main__": main() 