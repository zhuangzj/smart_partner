# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:26:07 2017

@author: zijunzhuang
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
chinesefont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')

def main():
    math_df = pd.read_excel('./data/tongzhou_201701/math.xlsx')
    print(math_df.head())
    print('\n')
    print(math_df.shape)
    math_df['TotalScore'] = math_df.iloc[:, 5:32].sum(axis=1)
    ranges = np.arange(0,110,10).tolist()
    math_df['ScoreGrp'] = pd.cut(math_df['TotalScore'], bins = ranges)
    for school_group, school_frame in math_df.groupby('学校'):
        data = []
        index = []
        for group, frame in school_frame.groupby('班级'):
            index.append(group)
            row = pd.value_counts(frame['ScoreGrp'])
            data.append(row)
        math_df = pd.DataFrame(data)
        math_df = math_df.set_index([index])
        print(math_df)
        ax = math_df.plot(kind='bar', figsize=(15.5, 7.5))
        ax.set_xticklabels(math_df.index.values, fontproperties=chinesefont)
        ax.set_title(school_group, fontproperties=chinesefont)
        plt.show()
        
if __name__ == "__main__": main() 