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
    excel_r = len(df[df['总分'] > 85]) / len(df['总分'])   
    pass_r = len(df[df['总分'] > 60]) / len(df['总分'])
    progress_r = df['进步幅度'].sum() / len(df['总分'])        
    return excel_r, pass_r, progress_r    

def find_rank_by_score(score, score_list):
    
    return score_list.index(score) + 1

def add_school_rank(df):
    frames = []
    # rank by score
    for school_group, school_frame in df.groupby('学校'):
        # get unique score and then sort in order to rank by score
        uniq_score = school_frame['总分'].unique()
        score_list = pd.Series(uniq_score).sort_values(ascending = False).tolist()
        school_frame['学校排名'] = school_frame['总分'].apply(find_rank_by_score, args=(score_list,))
        #print(school_frame)
        frames.append(school_frame)
        
    df = pd.concat(frames)
    return df
    
def main():
    df_01 = pd.read_excel('./data/tongzhou_201701/math.xlsx')
    df_05 = pd.read_excel('./data/tongzhou_201705/math.xls', skiprows=1)
    
    # each stu's total score of the exam
    df_01['总分'] = df_01.iloc[:, 5:32].sum(axis=1)
    # give name to the title in loc 31
    df_05.columns.values[31] = '总分'
    

    
    # merge o1 data to get the school and class info
    df_05.rename(columns = {'教育ID' : '学号'}, inplace = True)
    df_05 = df_05[['学号', '总分']].merge(df_01[['学校', '班级', '学号', '姓名', '性别']], on = '学号')
    
    # check: id same, name different
    #print(df_05[df_05['姓名_x'] != df_05['姓名_y']][['姓名_x', '姓名_y']])
    
    # rank by score
    df_01 = add_school_rank(df_01)
    df_05 = add_school_rank(df_05)
    df_01.rename(columns = {'学校排名' : '201701学校排名', '总分' : '201701总分'}, inplace = True)
    #df_05.rename(columns = {'学校排名' : '201705学校排名', '总分' : '201705总分'}, inplace = True)
    merged_df = df_05.merge(df_01[['学号', '201701总分', '201701学校排名']], on = '学号')
    merged_df['进步幅度'] = (merged_df['201701学校排名'] - merged_df['学校排名']) / merged_df['201701学校排名']
    merged_df.to_csv('201701merge201705.csv')
#    print(df)
#    print(df.columns)
    
    data = []
    index = []
    for school_group, school_frame in merged_df.groupby('学校'):       
        data = []
        index = []
        for group, frame in school_frame.groupby('班级'):
            excel_r, pass_r, progress_r = class_state_ratio(frame)
            index.append(group)
            data.append({'excel' : excel_r, 'pass' : pass_r, 'progress' : progress_r})
        
        df = pd.DataFrame(data)
        df.set_index([index], inplace = True)
        ax = df.plot(kind='bar',ylim = (-1, 1), figsize=(15.5, 7.5))
        ax.set_xticklabels(df.index.values, fontproperties=chinesefont)
        ax.set_title(school_group, fontproperties=chinesefont)
        plt.show()
        
if __name__ == "__main__": main()