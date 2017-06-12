# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:57:47 2017

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
    
    D_list = [calcul_D(chinese_df, 100), calcul_D(math_df, 100), calcul_D(english_df, 65), calcul_D(physics_df, 100), calcul_D(biology_df, 100), calcul_D(history_df, 100), calcul_D(ethic_df, 100)]
    distic_df = pd.DataFrame(data = [D_list], columns = ['chinese', 'math', 'english', 'physics', 'biology', 'history', 'ethic'])
    print(distic_df)
    
   
    
def calcul_D(df, W):
    # sort on subject's score
    df = df.sort_values('TotalScore', ascending = False)
    
    XH = df.head(int(len(df.index)*0.27))['TotalScore'].mean()
    #print(XH)
    XL = df.tail(int(len(df.index)*0.27))['TotalScore'].mean()
    #print(XL)
    D = 2 * (XH - XL) / W # W is total score of a test of a question
    #print(D)
    return D
    
if __name__ == "__main__": main() 