#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:59:27 2020

@author: xue
"""
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'/home/xue/stanford-corenlp-full-2018-10-05')
f_train_token = open("input_data/train_token.txt","r")
f_test_token = open("input_data/test_token.txt","r")
f_train_pos = open("input_data/train_pos.txt","w")
f_test_pos = open("input_data/test_pos.txt","w")

def producePostag(fin,f_pos_tag):
    f = fin
    fout = f_pos_tag
    while(True):
        split_tokens = f.readline().split()
     #   print (split_tokens.length)
        if len(split_tokens) == 0:break
        char_tokens = " ".join(split_tokens)
        pos_tag = nlp.pos_tag(char_tokens)
        for pos in pos_tag:
            fout.write(pos[1])
            fout.write(" ")
        fout.write("\n")
    
producePostag(f_train_token,f_train_pos)
producePostag(f_test_token,f_test_pos)