# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:16:13 2017

@author: zhuangzijun
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
chinesefont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')

def class_state_ratio(df):
    excel_r = len(df[df['TotalScore'] > 85]) / len(df['TotalScore'])   
    pass_r = len(df[df['TotalScore'] > 60]) / len(df['TotalScore'])        
    return excel_r, pass_r    

def main():
    df_01 = pd.read_excel('./data/tongzhou_201701/math.xlsx')
    df_05 = pd.read_excel('./data/tongzhou_201705/math.xls', skiprows=1)
    # give name to the title in loc 31
    df_05.columns.values[31] = 'TotalScore'
    # merge o1 data to get the school and class info
    df_05.rename(columns = {'教育ID' : '学号'}, inplace = True)
    df_05 = df_05.merge(df_01.iloc[:, 0:5], on = '学号')
    # check: id same, name different
    #print(df_05[df_05['姓名_x'] != df_05['姓名_y']][['姓名_x', '姓名_y']])
    print(df_05.columns)
    data = []
    index = []
    for school_group, school_frame in df_05.groupby('学校'):
        data = []
        index = []
        for group, frame in school_frame.groupby('班级'):
            excel_r, pass_r = class_state_ratio(frame)
            index.append(group)
            data.append({'excel' : excel_r, 'pass' : pass_r})
        
        df = pd.DataFrame(data)
        df.set_index([index], inplace = True)
        ax = df.plot(kind='bar', figsize=(15.5, 7.5))
        ax.set_xticklabels(df.index.values, fontproperties=chinesefont)
        ax.set_title(school_group, fontproperties=chinesefont)
        plt.show()
        
if __name__ == "__main__": main()