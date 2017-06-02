# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:23:31 2017

@author: zhuangzijun
"""

import pandas as pd
import numpy as np

def main():
    df = pd.read_excel('./data/recommend.xlsx', skiprows=1)
       
    for index, row in df.iterrows():
        recommd_201701_55_count = 0
        recommd_201701_46_count = 0
        #if 55 and 46 in 201701_recommend
        for a in row[19:22]:
            if a != np.nan:
                if a in row[5:9].values:
                    recommd_201701_55_count+=1
        for b in row[24:27]:
            if b != np.nan:
                if b in row[5:9].values:
                    recommd_201701_46_count+=1
                    
        recommd_201705_55_count = 0
        recommd_201705_46_count = 0
        #if 55 and 46 in 201705_recommend
        for c in row[19:22]:
            if c != np.nan:
                if c in row[11:17].values:
                    recommd_201705_55_count+=1
        for d in row[24:27]:
            if d != np.nan:
                if d in row[11:17].values:
                    recommd_201705_46_count+=1
                    
        df.loc[index, '201701_55'] = recommd_201701_55_count
        df.loc[index, '201701_46'] = recommd_201701_46_count
        df.loc[index, '201705_55'] = recommd_201705_55_count                 
        df.loc[index, '201705_46'] = recommd_201705_46_count

    df.to_csv('./data/diff_recommend3.csv')

    
    
if __name__ == "__main__": main()