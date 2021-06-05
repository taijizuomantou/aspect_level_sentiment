#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:28:49 2019

@author: xue
"""
import argparse
import collections

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

def get_y_true(task_name):
    """ 
    Read file to obtain y_true.
    All of five tasks of Sentihood use the test set of task-BERT-pair-NLI-M to get true labels.
    All of five tasks of SemEval-2014 use the test set of task-BERT-pair-NLI-M to get true labels.
    """
    if task_name in ["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M", "sentihood_NLI_B", "sentihood_QA_B"]:
        true_data_file = "data/sentihood/bert-pair/test_NLI_M.tsv"

        df = pd.read_csv(true_data_file,sep='\t')
        y_true = []
        for i in range(len(df)):
            label = df['label'][i]
            assert label in ['None', 'Positive', 'Negative'], "error!"
            if label == 'None':
                n = 0
            elif label == 'Positive':
                n = 1
            else:
                n = 2
            y_true.append(n)
    else:
        true_data_file = "data/semeval2014/bert-pair/test_NLI_M.csv"

        df = pd.read_csv(true_data_file,sep='\t',header=None).values
        y_true=[]
        for i in range(len(df)):
            label = df[i][1]
            assert label in ['positive', 'neutral', 'negative', 'conflict', 'none'], "error!"
            if label == 'positive':
                n = 0
            elif label == 'neutral':
                n = 1
            elif label == 'negative':
                n = 2
            elif label == 'conflict':
                n = 3
            elif label == 'none':
                n = 4
            y_true.append(n)
    
    return y_true

def semeval_PRF(y_true, y_pred,score,detect_score):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    
    s_all=0
    g_all=0
    s_g_all=0
    for i in range(len(y_pred)//5):
        s=set()
        g=set()
        for j in range(5):
            if y_pred[i*5+j]!=0:
                s.add(j)
            else:
                maximum = 0
                for idd,item in enumerate(score[i*5+j]):
                    if item > maximum : maximum = item
                if maximum > 0.8:
                    pass
                   # s.add(j)
                    
            if y_true[i*5+j]!=4:
                g.add(j)
        if len(g)==0:continue
        s_g=s.intersection(g)
        s_all+=len(s)
        g_all+=len(g)
        s_g_all+=len(s_g)

    p=s_g_all/s_all
    r=s_g_all/g_all
    f=2*p*r/(p+r)
#    fin = open("data/semeval2014/bert-pair/test_NLI_M.csv","r")
#    whole_file = []
#    for line in fin:
#        item = line.split("\t")
#        whole_file.append(item)
#    file = open("temp.txt","w")
#    #ans = []
#    ans = 0
#    whole_wrong = 0
#    count = 0
#    for i in range(len(y_true)):
#        if y_true[i] != 4:
#            y_true[i] = 1
#        else:
#            y_true[i] = 0
#        if y_true[i] != y_pred[i]:whole_wrong += 1
#        if y_true[i] != y_pred[i] and y_pred[i] == 0 :#and y_pred[i] == 0:
#           # print(i)
##           begin = int(i / 5 * 5)
##           count = 0
##          # file.write(str(whole_file[i][2]))
##           for j in range(begin,begin+5):
##               if whole_file[j][1] != 'none':count += 1
##           if count > 1 : 
##               ans += 1
#           file.write(str(y_pred[i]))
#           file.write("    ")
#           file.write(str(whole_file[i][1]))
#           file.write("    ")
#           file.write(str(whole_file[i][2]))
#           file.write("    ")
#           file.write(str(whole_file[i][3]))
#           # file.write(str(detect_score[i]))
#           #file.write(str(score[i]))
#            #file.write("\n")
#        
#    print(ans)
#    print(whole_wrong)
    return p,r,f


def semeval_Acc(y_true, y_pred, score, classes=4):
    """
    Calculate "Acc" of sentiment classification task of SemEval-2014.
    """
    assert classes in [2, 3, 4], "classes must be 2 or 3 or 4."

    if classes == 4:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]==4:continue
            total+=1
            tmp=y_pred[i]
            if tmp==4:
                if score[i][0]>=score[i][1] and score[i][0]>=score[i][2] and score[i][0]>=score[i][3]:
                    tmp=0
                elif score[i][1]>=score[i][0] and score[i][1]>=score[i][2] and score[i][1]>=score[i][3]:
                    tmp=1
                elif score[i][2]>=score[i][0] and score[i][2]>=score[i][1] and score[i][2]>=score[i][3]:
                    tmp=2
                else:
                    tmp=3
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total
    elif classes == 3:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]>=3:continue
            total+=1
            tmp=y_pred[i]
            if tmp>=3:
                if score[i][0]>=score[i][1] and score[i][0]>=score[i][2]:
                    tmp=0
                elif score[i][1]>=score[i][0] and score[i][1]>=score[i][2]:
                    tmp=1
                else:
                    tmp=2
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total
    else:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]>=3 or y_true[i]==1:continue
            total+=1
            tmp=y_pred[i]
            if tmp>=3 or tmp==1:
                if score[i][0]>=score[i][2]:
                    tmp=0
                else:
                    tmp=2
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total

    return sentiment_Acc
pred_data_dir = "results/sentihood/NLI_M/test_ep_4.txt"
#pred_data_dir = "results(2)/48_remove_aspect_embedding_weight/NLI_M/test_ep_6.txt"  
detect_pred=[]
y_pred = []
score=[]
detect_score=[]
with open(pred_data_dir,"r",encoding="utf-8") as f:
    s=f.readline().strip().split()
    i = 0
    while s:
        if i % 16 < 8:
            detect_pred.append(int(s[0]))
            detect_score.append([float(s[1]), float(s[2])])
        else:
            y_pred.append(int(s[0]))
            score.append([float(s[1]), float(s[2]), float(s[3]), float(s[4])])
        s = f.readline().strip().split()
        i += 1
y_true = get_y_true("semeval_QA_M")
aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, detect_pred, score,detect_score)
sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
result = {'aspect_P': aspect_P,
        'aspect_R': aspect_R,
        'aspect_F': aspect_F,
        'sentiment_Acc_4_classes': sentiment_Acc_4_classes,
        'sentiment_Acc_3_classes': sentiment_Acc_3_classes,
        'sentiment_Acc_2_classes': sentiment_Acc_2_classes}

for key in result.keys():
    print(key, "=",str(result[key]))
