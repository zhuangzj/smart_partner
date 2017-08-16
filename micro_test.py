# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 10:29:35 2017

@author: zhuangzijun
"""

import pandas as pd
import re

def overall_test(subject, suffix):
    df01 = pd.read_excel('./data/subject_literacy/grade8-'+subject+'201701.'+suffix, converters={'教育ID':str})
    df05 = pd.read_excel('./data/subject_literacy/grade8-'+subject+'201705.'+suffix, converters={'教育ID':str})
    df07 = pd.read_excel('./data/subject_literacy/grade8-'+subject+'201707.'+suffix, converters={'教育ID':str})
    coding01 = pd.read_excel('./data/subject_literacy/201701-初二数学编码.xlsx')
    coding05 = pd.read_excel('./data/subject_literacy/201705-初二数学编码.xlsx')
    coding07 = pd.read_excel('./data/subject_literacy/201707-初二数学编码.xlsx')
    # 将最后三个编码从数字转成文本
    df01.columns.values[28:31] = list(map(lambda x: '0'+str(x), df01.columns[28:31]))
    df01.columns.values[66:69] = list(map(lambda x: '0'+str(x), df01.columns[66:69]))
    # 学生各题得分
    df01 = get_overall_test_score(0, 31, df01)
    coding01 = get_overall_test_code(coding01)
    df05 = get_overall_test_score(0, 29, df05)
    coding05 = get_overall_test_code(coding05)
    df07 = get_overall_test_score(0, 38, df07)
    coding07 = get_overall_test_code(coding07)
    #df05, coding05 = tidy_overall_test_and_coding(3, 29, 39, 65, df05)
    #df07, coding07 = tidy_overall_test_and_coding(3, 38, 48, 83, df07)
    return df01, coding01, df05, coding05, df07, coding07

def get_overall_test_score(start, end, df):
    df = df.iloc[2:, start:end]
    df.drop('题目编码', axis=1, inplace=True)
    df.set_index(['教育ID', '姓名'], inplace=True)
    return df

def get_overall_test_code(coding):
    coding = coding.iloc[1:, :]
    coding['题目编码'] = '0' + coding['题目编码'].astype('int').astype(str)
    coding = coding[['题目编码', '学习表现指标代码', '核心概念', '核心素养', '评分标准']]
    coding['评分标准'] = coding['评分标准'].apply(lambda x: get_score(x))
    coding.rename(columns={'评分标准': '总分'}, inplace=True)
    coding.set_index(['题目编码'], inplace=True)
    return coding

def tidy_overall_test_and_coding(ability_start, ability_end, suyang_start, suyang_end, df):
    # 素养上的编码去掉.1
    df.columns.values[suyang_start:suyang_end] = list(map(lambda x: x[:-2], df.columns[suyang_start:suyang_end]))
    ability = df.iloc[0, ability_start:ability_end]
    ability.name='学习表现指标'
    suyang = df.iloc[0, suyang_start:suyang_end]
    suyang.name='核心素养'
    coding = pd.concat([ability, suyang], axis=1)
    df = df.iloc[2:, 0:ability_end]
    #df.set_index(['教育ID', '姓名'], inplace=True)
    df.drop('题目编码', axis=1, inplace=True)
    return df, coding
    
def micro_test():
    df1, coding1 = read_excel_by_name('2016-数学-八年级-上学期-单元微测-001（二次根式1）')
    df2, coding2 = read_excel_by_name('2016-数学-八年级-上学期-单元微测-002（二次根式2）')
    df3, coding3 = read_excel_by_name('2016-数学-八年级-上学期-单元微测-001（分式1）')
    df4, coding4 = read_excel_by_name('2016-数学-八年级-上学期-单元微测-002（分式2）')
    df5, coding5 = read_excel_by_name('2016-数学-八年级-下学期-单元微测-001（变量之间的关系）')
    df1, coding1 = tidy_micro_coding_A(df1, coding1)
    df2, coding2 = tidy_micro_coding_A(df2, coding2)
    df3, coding3 = tidy_micro_coding_B(df3, coding3)
    df4, coding4 = tidy_micro_coding_B(df4, coding4)
    df5, coding5 = tidy_micro_coding_B(df5, coding5)
    df1.set_index(['教育ID', '姓名'], inplace=True)
    df2.set_index(['教育ID', '姓名'], inplace=True)
    df3.set_index(['教育ID', '姓名'], inplace=True)
    df4.set_index(['教育ID', '姓名'], inplace=True)
    df5.set_index(['教育ID', '姓名'], inplace=True)
    return df1, coding1, df2, coding2, df3, coding3, df4, coding4, df5, coding5

def read_excel_by_name(filename):
    df = pd.read_excel('./data/micro_test/' + filename + '.xlsx', sheetname=1, converters={'学号':str})
    coding = pd.read_excel('./data/micro_test/' + filename + '.xlsx', sheetname=2)
    return df, coding

def tidy_micro_coding_A(df, coding):
    coding = coding.iloc[1:]
    coding['题目编码'] = '0' + coding['题目编码'].astype('int').astype(str)
    coding['题目编码'] = coding['题目编码'].apply(lambda x: 'P'+x[:-2]+'01'+x[-2:])
    coding['评分标准'] = coding['评分标准'].apply(lambda x: get_score(x))
    coding.columns.values[1] = '学习表现指标'
    coding.columns.values[3] = '总分'
    coding.set_index(['题目编码'], inplace=True)
    df = tidy_micro_test(df)
    return df, coding

def tidy_micro_coding_B(df, coding):
    # cause kernel die
    # knowledge = list(coding.iloc[0, 2:8])
    # context = list(coding.iloc[0, 12:18])
    # coding.columns.values[2:8] = knowledge
    # coding.columns.values[12:18] = context
    coding = coding.iloc[1:, :]
    coding['题目编码'] = 'P0' + coding['题目编码'].astype('int').astype(str) + '01'
    coding = coding[['题目编码', '学习表现指标代码', '核心素养', '评分标准', '能力水平']]
    coding['评分标准'] = coding['评分标准'].apply(lambda x: get_score(x))
    coding.columns.values[1] = '学习表现指标'
    coding.columns.values[3] = '总分'
    coding.set_index(['题目编码'], inplace=True)
    df = tidy_micro_test(df)
    return df, coding
 
def tidy_micro_test(df):
    df = df.iloc[:, 6:]
    df.drop('性别', axis=1, inplace=True)
    df.rename(columns={'学号': '教育ID'}, inplace=True)
    # cannot merge overall and micro on index
    #df.set_index(['教育ID', '姓名'], inplace=True) 
    return df

def get_score(x):
    score = x[x.index('【'):x.index('】')+1]  
    comma = re.search(',|，', score)
    # 百分比总分
    score = score.split(comma.group(0))[1] 
    score = score[0:score.index('分')]
    return score

def overlap_df(overall01, overall05, overall07, micro1, micro2, micro3, micro4, micro5):
    
    micro1_overlap = overlap(overall01, overall05, overall07, micro1)
    micro2_overlap = overlap(overall01, overall05, overall07, micro2)
    micro3_overlap = overlap(overall01, overall05, overall07, micro3)
    micro4_overlap = overlap(overall01, overall05, overall07, micro4)
    micro5_overlap = overlap(overall01, overall05, overall07, micro5)
    df = pd.DataFrame({'2016-数学-八年级-上学期-单元微测-001（二次根式1）': micro1_overlap, 
                  '2016-数学-八年级-上学期-单元微测-002（二次根式2）': micro2_overlap,
                  '2016-数学-八年级-上学期-单元微测-001（分式1）': micro3_overlap,
                  '2016-数学-八年级-上学期-单元微测-002（分式2）': micro4_overlap,
                  '2016-数学-八年级-下学期-单元微测-001（变量之间的关系）': micro5_overlap}, index=['201701', '201705', '201707'])
    return df

def overlap(overall01, overall05, overall07, micro):
    micro_overlap = []
    micro_overlap.append(pd.merge(overall01, micro, on=['教育ID', '姓名']).shape[0])
    micro_overlap.append(pd.merge(overall05, micro, on=['教育ID', '姓名']).shape[0])
    micro_overlap.append(pd.merge(overall07, micro, on=['教育ID', '姓名']).shape[0])
    return micro_overlap

def overall_test_avg_score(score_df, code_df):
    #print(code_df)
    concepts = []
    for colname, col in score_df.iteritems():
        score = code_df.loc[colname]['总分']
        concept = code_df.loc[colname]['核心概念']
        concepts.append(concept)
        score_df[colname] = score_df[colname].map(lambda x: x/int(score))
        #score_df.rename(columns={colname: concept}, inplace=True)
        #print(list(filter(lambda x: x == colname, code_df.index.tolist())))
   
    score_df.columns = concepts    
    # group into each concept
    grouped = score_df.groupby(score_df.columns, axis=1)
    return grouped.mean()
    
def micro_test_avg_score(score_df, code_df):
    for colname, col in score_df.iteritems():
        score = code_df.loc[colname]['总分']
        score_df[colname] = score_df[colname].map(lambda x: x/int(score))  
    return score_df.mean(axis=1)

def main():
    overall01, coding01, overall05, coding05, overall07, coding07 = overall_test('math', 'xls')
    avg_overall01 = overall_test_avg_score(overall01, coding01)
    #overall_test_avg_score(overall05, coding05)
    #overall_test_avg_score(overall07, coding07)
    
    micro1, coding1, micro2, coding2, micro3, coding3, micro4, coding4, micro5, coding5 = micro_test()
    fraction_micro1 = micro_test_avg_score(micro1, coding1)
    fraction_micro1.name = '数学单元微测-001（分式1）'
    fraction_micro2 = micro_test_avg_score(micro2, coding2)
    fraction_micro2.name = '数学单元微测-002（分式2）'
    quadratic_micro1 = micro_test_avg_score(micro3, coding3)
    quadratic_micro1.name = '数学单元微测-001（二次根式1）'
    quadratic_micro2 = micro_test_avg_score(micro4, coding4)
    quadratic_micro2.name = '数学单元微测-002（二次根式2）'
    
    # 201701总测和2次分式微测
    fraction_overall = avg_overall01['分式']
    fraction_overall.name = '201701数学总测'
    fraction_df = pd.concat([fraction_overall, fraction_micro1, fraction_micro2], axis=1)
    print(fraction_df)
    fraction_df.to_csv('./data/output/分式.csv')
    
    # 201701总测和2次二次根式微测
    quadratic_overall = avg_overall01['二次根式']
    quadratic_overall.name = '201701数学总测'
    quadratic_df = pd.concat([quadratic_overall, quadratic_micro1, quadratic_micro2], axis=1)
    print(quadratic_df)
    quadratic_df.to_csv('./data/output/二次根式.csv')
    #score = coding1['总分'] 
    #for i, s in enumerate(score.tolist()):
        #micro1.iloc[:, i].divide(s) #micro1.loc[colname] = micro1.loc[colname] / score 
        
#    df = overlap_df(overall01, overall05, overall07, micro1, micro2, micro3, micro4, micro5)
#    df.to_csv('./data/总测微测.csv')
    
if __name__ == "__main__": main() 

