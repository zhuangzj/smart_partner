# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:25:00 2017

@author: zhuangzijun
"""

import pandas as pd

def main():
    chinese_df = pd.read_excel('./data/tongzhou_201701/chinese.xlsx')
    math_df = pd.read_excel('./data/tongzhou_201701/math.xlsx')
    english_df = pd.read_excel('./data/tongzhou_201701/english.xlsx')
    physics_df = pd.read_excel('./data/tongzhou_201701/physics.xlsx')
    biology_df = pd.read_excel('./data/tongzhou_201701/biology.xlsx')
    history_df = pd.read_excel('./data/tongzhou_201701/history.xlsx')
    ethic_df = pd.read_excel('./data/tongzhou_201701/ethic.xlsx')

    chinese_df['TotalScore'] = chinese_df.iloc[:, 5:47].sum(axis=1)
    math_df['TotalScore'] = math_df.iloc[:, 5:32].sum(axis=1)
    english_df['TotalScore'] = english_df.iloc[:, 5:40].sum(axis=1)
    physics_df['TotalScore'] = physics_df.iloc[:, 5:53].sum(axis=1)
    biology_df['TotalScore'] = biology_df.iloc[:, 5:46].sum(axis=1)
    history_df['TotalScore'] = history_df.iloc[:, 5:36].sum(axis=1)
    ethic_df['TotalScore'] = ethic_df.iloc[:, 5:41].sum(axis=1)
    
    
    cols = ['学校', '班级', '学号', '姓名', '性别', 'TotalScore']
    chinese_df = chinese_df[cols]
    math_df = math_df[cols]
    english_df = english_df[cols]
    physics_df = physics_df[cols]
    biology_df = biology_df[cols]
    history_df = history_df[cols]
    ethic_df = ethic_df[cols]
    
    # rename TotaScore of each subject
    chinese_df = chinese_df.rename(columns = {'TotalScore':'chinese'})
    math_df = math_df.rename(columns = {'TotalScore':'math'})
    english_df = english_df.rename(columns = {'TotalScore':'english'})
    physics_df = physics_df.rename(columns = {'TotalScore':'physics'})
    biology_df = biology_df.rename(columns = {'TotalScore':'biology'})
    history_df = history_df.rename(columns = {'TotalScore':'history'})
    ethic_df = ethic_df.rename(columns = {'TotalScore':'ethic'})
    
    # merge score on stu_id
    stu_id = ['学校', '班级', '学号', '姓名', '性别']
    df = chinese_df.merge(math_df, on=stu_id).merge(english_df, on=stu_id).merge(physics_df, on=stu_id).merge(biology_df, on=stu_id).merge(history_df, on=stu_id).merge(ethic_df, on=stu_id)
    #print(df.head())
    
    # find the columns that have NaN values
    #print(df.isnull().any())
    
    df = df.iloc[:, 5:12]
    # normalization
    df[['chinese', 'math', 'physics', 'biology', 'history', 'ethic']] = df[['chinese', 'math', 'physics', 'biology', 'history', 'ethic']]/100
    df['english'] = df['english']/65
    print(df.corr(method='pearson'))
    
    
if __name__ == "__main__": main() 