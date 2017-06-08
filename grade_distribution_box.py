# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:35:10 2017

@author: zijunzhuang
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
chinesefont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')

def main():
    math_df = pd.read_excel('./data/tongzhou_201701/math.xlsx')
   
    math_df['TotalScore'] = math_df.iloc[:, 5:32].sum(axis=1)
    for school_group, school_frame in math_df.groupby('学校'):
        class_df = school_frame[['班级', '学号', 'TotalScore']]
        class_df = class_df.pivot(index='学号', columns='班级', values='TotalScore')
        ax = class_df.plot(kind='box', figsize=(15.5, 7.5))
        ax.set_xticklabels(class_df.columns.values, fontproperties=chinesefont)
        ax.set_title(school_group, fontproperties=chinesefont)
        plt.show()
        
if __name__ == "__main__": main() 