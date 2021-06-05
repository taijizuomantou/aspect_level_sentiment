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
    true_data_file = "data/sentihood/bert-pair/test_NLI_M.tsv"
    df = pd.read_csv(true_data_file,sep='\t')
    y_true = []
    for i in range(len(df)):
        label = df['label'][i]
        assert label in [ 'Positive', 'Negative','None'], "error!"
        if label == 'None':
            n = 2
        elif label == 'Positive':
            n = 0
        elif label == "Negative":
            n = 1
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!")
        y_true.append(n)
    return y_true

def get_y_yuan(task_name):
    """ 
    Read file to obtain y_true.
    All of five tasks of Sentihood use the test set of task-BERT-pair-NLI-M to get true labels.
    All of five tasks of SemEval-2014 use the test set of task-BERT-pair-NLI-M to get true labels.
    """
    true_data_file = "data/sentihood/bert-pair/test_NLI_M.tsv"
    df = pd.read_csv(true_data_file,sep='\t')
    y_true = []
    for i in range(len(df)):
        label = df['label'][i]
        assert label in [ 'Positive', 'Negative','None'], "error!"
        if label == 'None':
            n = 0
        elif label == 'Positive':
            n = 1
        else:
            n = 2
        y_true.append(n)
    return y_true



def sentihood_strict_acc(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of Sentihood.
    """
    my_y_true = []
    for item in y_true:
        if item == 2:
            my_y_true.append(0)
        else:
            my_y_true.append(1)
    total_cases=int(len(y_true)/4)
    true_cases=0
    for i in range(total_cases):
        if my_y_true[i*4]!=y_pred[i*4]:continue
        if my_y_true[i*4+1]!=y_pred[i*4+1]:continue
        if my_y_true[i*4+2]!=y_pred[i*4+2]:continue
        if my_y_true[i*4+3]!=y_pred[i*4+3]:continue
        true_cases+=1
    aspect_strict_Acc = true_cases/total_cases
    return aspect_strict_Acc


def sentihood_macro_F1(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of Sentihood.
    """
    fin = open("data/sentihood/bert-pair/test_NLI_M.tsv","r")
    whole_file = []
    for line in fin:
        item = line.split("\t")
        whole_file.append(item)
    f = open("temp.txt","w")
    my_y_true = []
    for item in y_true:
        
        if item == 2:
            my_y_true.append(0)
        else:
            my_y_true.append(1)
#    for i in range(len(my_y_true)):
#        if my_y_true[i] == 0:
#            f.write(str(whole_file[i+1][3]))
           # f.write("\n")
    
    
    for i in range(int(len(y_pred)/4)):
        count = 0
        named = -1
        for j in range(4):
            
            idd = i * 4 + j
            if my_y_true[idd] == 1:
                count += 1
                named = idd
        for j in range(4):
            idd = i * 4 + j
            if y_pred[idd] != my_y_true[idd] and my_y_true[idd] == 0 and count == 1:
                f.write(str(idd))
                f.write("    ")
                f.write(str(count))
                f.write("    ")
                f.write(str(whole_file[idd+1][3][0]))
                f.write(str(whole_file[idd+1][3][1]))
                f.write("    ")
                f.write(str(whole_file[idd+1][2]))
                f.write("    ")
              #  f.write(str(whole_file[idd+1][1]))
                
                f.write("\n")
                f.write(str(whole_file[named+1][3][0]))
                f.write(str(whole_file[named+1][3][1]))
                f.write("    ")
                f.write(str(whole_file[named+1][2]))
                f.write("    ")
                f.write(str(whole_file[named+1][1]))
                f.write("\n")
    p_all=0
    r_all=0
    count=0
    print(len(y_true))
    for i in range(len(y_pred)//4):
        a=set()
        b=set()
        for j in range(4):
            if y_pred[i*4+j]!=0:
                a.add(j)
            if my_y_true[i*4+j]!=0:
                b.add(j)
        if len(b)==0:continue
        a_b=a.intersection(b)
        if len(a_b)>0:
            p=len(a_b)/len(a)
            r=len(a_b)/len(b)
          # print(p)
           #print(r)
        else:
            p=0
            r=0
        count+=1
        p_all+=p
        r_all+=r
    Ma_p=p_all/count
    Ma_r=r_all/count
    #print(Ma_p)
    aspect_Macro_F1 = 2*Ma_p*Ma_r/(Ma_p+Ma_r)
   # print()
    print(Ma_p)
    print(Ma_r)
    return aspect_Macro_F1


def sentihood_detect_auc(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of Sentihood.
    Calculate "Acc" of sentiment classification task of Sentihood.
    """
    # aspect-Macro-AUC
    aspect_y_true=[]
    aspect_y_score=[]
    aspect_y_trues=[[],[],[],[]]
    aspect_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]!=0:
            aspect_y_true.append(0)
        else:
            aspect_y_true.append(1) # "None": 1
        tmp_score=score[i][0] # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i%4].append(aspect_y_true[-1])
        aspect_y_scores[i%4].append(aspect_y_score[-1])

    aspect_auc=[]
    for i in range(4):
        aspect_auc.append(metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i]))
    aspect_Macro_AUC = np.mean(aspect_auc)
    return aspect_Macro_AUC
def sentihood_senti_auc(y_true, score):   
    # sentiment-Macro-AUC
    sentiment_y_true=[]
    sentiment_y_pred=[]
    sentiment_y_score=[]
    sentiment_y_trues=[[],[],[],[]]
    sentiment_y_scores=[[],[],[],[]]
    for i in range(len(y_true)):
        if y_true[i]>0:
            sentiment_y_true.append(y_true[i]-1) # "Postive":0, "Negative":1
            tmp_score=score[i][1]/(score[i][0]+score[i][1])  # probability of "Negative"
            sentiment_y_score.append(tmp_score)
            if tmp_score>0.5:
                sentiment_y_pred.append(1) # "Negative": 1
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i%4].append(sentiment_y_true[-1])
            sentiment_y_scores[i%4].append(sentiment_y_score[-1])

    sentiment_auc=[]
    for i in range(4):
        sentiment_auc.append(metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i]))
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    return  sentiment_Acc, sentiment_Macro_AUC





    
pred_data_dir = "senti_results/35_base/NLI_M/test_ep_4.txt"
detect_pred=[]
y_pred = []
score=[]
detect_score=[]
with open(pred_data_dir,"r",encoding="utf-8") as f:
    s=f.readline().strip().split()
    i = 0
    while s:
        if i % 8 < 4:
           # print("<16"+str(i))
            detect_pred.append(int(s[0]))
            detect_score.append([float(s[1]), float(s[2])])
        else:
          #  print(i)
            y_pred.append(int(s[0]))
            score.append([float(s[1]), float(s[2])])
        s = f.readline().strip().split()
        i += 1
print(len(y_pred))
y_true = get_y_true("semeval_QA_M")
y_yuan = get_y_yuan("")
aspect_Macro_F1 = sentihood_macro_F1(y_true, detect_pred)
acc = sentihood_strict_acc(y_true,detect_pred)
detect_auc = sentihood_detect_auc(y_yuan,detect_score)
senti_acc,senti_auc  = sentihood_senti_auc(y_yuan,score)
#sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
#sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
#sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
result = {'acc':acc,
        'aspect_Macro_F1': aspect_Macro_F1,
             
             "detect_auc":detect_auc,
             "senti_acc":senti_acc,
             'senti_auc':senti_auc
        #'sentiment_Acc_4_classes': sentiment_Acc_4_classes,
        #'sentiment_Acc_3_classes': sentiment_Acc_3_classes,
        #'sentiment_Acc_2_classes': sentiment_Acc_2_classes
        }
for key in result.keys():
    print(key, "=",str(result[key]))