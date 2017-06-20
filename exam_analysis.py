# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:22:21 2017

@author: zhuangzijun
"""
import pandas as pd

def difficulty(df):
    
    df.loc['试卷得分']['总分'] = 100
    df.dropna(inplace=True)
    df['难度'] = df['平均分'] / df['总分']
    print(df)
    return df 
    
def get_q_total_score():
    test_notation = pd.read_excel('./data/tongzhou_201701/test_notation.xlsx', skiprows=2,
                              names=['题目编码', '得分点编码', '一级主题代码', '一级主题', '二级主题代码', '二级主题', '核心概念代码', '核心概念', '学习表现指标代码', '学习表现指标', '核心素养代码', '核心素养',
                                     '内容属性代码', '内容属性', '熟悉度代码', '熟悉度', '间接度代码', '间接度', '评分标准', '总分'],
                                     converters={'题目编码':str,'得分点编码':str})
    #print(test_notation)
    test_notation['题目编号'] = 'P' + '0' + test_notation['题目编码'].map(str) + '0' + test_notation['得分点编码'].map(str)
    return test_notation[['题目编号', '总分']]


def distinction_extremgroup(df, q_score):
    df.sort_values('试卷得分', ascending = False, inplace = True)
    group_perc = 0.27
    XH = df.head(int(len(df.index) * group_perc))   
    XL = df.tail(int(len(df.index) * group_perc))
    # yes/no question
    XH_obj = XH[XH.iloc[:, 0:15] == 3].iloc[:, 0:15].count() / len(XH)
    XL_obj = XL[XL.iloc[:, 0:15] == 3].iloc[:, 0:15].count() / len(XL)
#    print(XH_obj)
#    print(XL_obj)
    obj_df = pd.DataFrame({'高分组': XH_obj, '低分组': XL_obj})
#    print(obj_df)
    
    # score point question
    XH_suj = XH.iloc[:, 15:29].mean()
    XL_suj = XL.iloc[:, 15:29].mean()
#    print(XH_suj)
#    print(XL_suj)
    suj_df = pd.DataFrame({'高分组': XH_suj, '低分组': XL_suj})
#    print(suj_df)
    concated_df = pd.concat([obj_df, suj_df])
    concated_df.reset_index(inplace=True)
    concated_df.columns.values[0] = '题目编号'
#    print(concated_df)
    
    df = concated_df.merge(q_score, on='题目编号', how='left')
    df.set_index(['题目编号'], inplace=True)
    df.loc['试卷得分']['总分'] = 100
    df.dropna(inplace=True)
    df['区分度'] = df['高分组'] - df['低分组']
    df.iloc[10:24, 3] = df.iloc[10:24, 3] / df.iloc[10:24, 2]
    return df
    
    
def main():
    math_df = pd.read_excel('./data/tongzhou_201701/math.xlsx')
    math_df['试卷得分'] = math_df.iloc[:, 5:32].sum(axis=1)
    math_df = math_df.iloc[:, 5:34]
    #print(math_df.columns)
    
    # q-mean
#    mean = math_df.mean()
#    q_mean = pd.DataFrame({'题目编号':mean.index, '平均分':mean.values})
    #print(q_mean)
    
    # q-totalscore
    q_score = get_q_total_score()
    #print(q_score)
    
#    dif_df = q_mean.merge(q_score, how='left', on='题目编号')
#    dif_df.set_index(['题目编号'], inplace=True)
    
#    dif_df = difficulty(dif_df)

    dis_df = distinction_extremgroup(math_df, q_score)
    print(dis_df)
    
if __name__ == "__main__": main() 