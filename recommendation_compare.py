# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:23:31 2017

@author: zhuangzijun
"""

import pandas as pd
import numpy as np

def main():
    cols = ['stuID', 'stuName', 'missExam', '201701_percentage', '201701_percentage_rank', '201701_recommend_a', '201701_recommend_b', '201701_recommend_c', '201701_recommend_d', 
                                            '201705_percentage', '201705_percentage_rank', '201705_recommend_a', '201705_recommend_b', '201705_recommend_c', '201705_recommend_d', '201705_recommend_e', '201705_recommend_f',
                                            '55_percentage', '55_percentage_rank', '55_recommend_a', '55_recommend_b', '55_recommend_c',
                                            '46_percentage', '46_percentage_rank', '46_recommend_a', '46_recommend_b', '46_recommend_c']
    #df = pd.read_excel('./data/recommend.xlsx', names=cols, skiprows=1)
    df = pd.read_excel('./data/recommend.xlsx', skiprows=1)
    output_cols = df.columns
    df.columns = cols
    #print(df.head(10))
    #print('\n')
    #print(df.shape)
    #print(df[df['201701_percentage'].isnull()])
    #print(df.iloc[0, 5:9])
    #a = df.iloc[1, 5:9]
    #b = df.iloc[1, 11:17]
    #c = df.iloc[1, 19:22]
    #d = df.iloc[1, 24:27]
    #print(a)
    #print(b)
    #print(c)
    #print(d)
    #print(type(a))
    #print(type(a.values))
    #print(''.join(a.values.tolist()))
    #a.replace(np.nan, '', regex=True, inplace=True)
    #str_a = ''.join(a.values.tolist())
    #b.replace(np.nan, '', regex=True, inplace=True)
    #str_b = ''.join(b.values.tolist())
    #print(str_a == str_b)
    #print(df.iloc[:, 6].isnull())
    #print(df.iloc[:, 5:9].head())
    #df[:, 5:9].replace(np.nan, '', regex=True, inplace=True)
    #print(df.iloc[:, 6].head())
    a_recommd_cols = ['201701_recommend_a', '201701_recommend_b', '201701_recommend_c', '201701_recommend_d']
    b_recommd_cols = ['201705_recommend_a', '201705_recommend_b', '201705_recommend_c', '201705_recommend_d', '201705_recommend_e', '201705_recommend_f']
    c_recommd_cols = ['55_recommend_a', '55_recommend_b', '55_recommend_c']
    d_recommd_cols = ['46_recommend_a', '46_recommend_b', '46_recommend_c']
    nan2emp_list = a_recommd_cols + b_recommd_cols + c_recommd_cols + d_recommd_cols
    df[nan2emp_list] = df[nan2emp_list].replace(np.nan, '', regex=True)
    diff_recommd_df = pd.DataFrame(columns=df.columns)
    #print(df.columns)
    for index, row in df.iterrows():
        a_recommd = row[5] + row[6] + row[7] + row[8]
        b_recommd = row[11] + row[12] + row[13] + row[14] + row[15] + row[16]
        c_recommd = row[19] + row[20] + row[21]
        d_recommd = row[24] + row[25] + row[26]
        if not a_recommd == b_recommd == c_recommd == d_recommd:
            diff_recommd_df.loc[len(diff_recommd_df)] = row
    #print(diff_recommd_df)
    diff_recommd_df.columns = output_cols
    diff_recommd_df.to_csv('./data/diff_recommend1.csv')
    #print(df[a_recommd_cols].head())
    #print(df[a_recommd_cols].values.tolist())
#    print(''.join(df[a_recommd_cols].values.tolist()))
#    temp_df = df[''.join(df[a_recommd_cols].values.tolist()) == 
#                 ''.join(df[b_recommd_cols].values.tolist()) == 
#                 ''.join(df[c_recommd_cols].values.tolist()) ==
#                 ''.join(df[d_recommd_cols].values.tolist())]
    
#    print(temp_df.head())
    
    
if __name__ == "__main__": main()