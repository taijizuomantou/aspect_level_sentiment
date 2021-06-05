#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:48:27 2020

@author: xue
"""


f_train_token = open("input_data/train_token.txt","r")
f_test_token = open("input_data/test_token.txt","r")
#f_train_pos = open("input_data/train_loc_1.txt","w")
#f_test_pos = open("input_data/test_loc_1.txt","w")
f_train_pos = open("input_data/train_loc_2.txt","w")
f_test_pos = open("input_data/test_loc_2.txt","w")

def produceLocPos(fin,f_pos_tag):
    f = fin
    fout = f_pos_tag
    while(True):
        split_tokens = f.readline().split()
     #   print (split_tokens.length)
        if len(split_tokens) == 0:break
        n = len(split_tokens)
        #position = [0] * n
        flag = False
        for i in range(0,n - 2):
            if split_tokens[i] == "location" and split_tokens[i + 1] == '-' and split_tokens[i + 2] == '2':
                fout.write(str(i))
                fout.write("\n")
                flag =True
                break
        if flag == False:
            fout.write("-1")
            fout.write("\n")
#            if split_tokens[i] == "location" and split_tokens[i + 1] == '-' and split_tokens[i + 2] == '2':
#                position[i] = 2
#                position[i + 1] = 2
#                position[i + 2] = 2
#        for item in position:
#            fout.write(str(item))
#            fout.write(" ")
#        fout.write("\n")
produceLocPos(f_train_token,f_train_pos)
produceLocPos(f_test_token,f_test_pos)