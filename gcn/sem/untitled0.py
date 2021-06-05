#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:20:33 2020

@author: xue
"""

fword = open("sem_test_stan_word.txt","r")
fhead = open("sem_test_stan_head.txt","r")
fout = open("sem_test_stan_true_head.txt","w")
while(1):
    words= fword.readline().split()
    if(len(words) == 0):break
    heads = fhead.readline().split()
    idx = []
    n = len(words)
    for i in range(0,n+1):
        idx.append(i)
    for i,word in enumerate(words):
        if(word[0] == "#"):
           for j in range(i + 1,n):
               idx[j] += 1
    num = 0
    fout.write("0 ")
    for head in heads:
        num = int(head)
        fout.write(str(idx[num]))
        fout.write(" ")
    fout.write("\n")

fword = open("sem_train_stan_word.txt","r")
fhead = open("sem_train_stan_head.txt","r")
fout = open("sem_train_stan_true_head.txt","w")
while(1):
    words= fword.readline().split()
    if(len(words) == 0):break
    heads = fhead.readline().split()
    idx = []
    n = len(words)
    for i in range(0,n+1):
        idx.append(i)
    for i,word in enumerate(words):
        if(word[0] == "#"):
           for j in range(i + 1,n):
               idx[j] += 1
    num = 0
    fout.write("0 ")
    for head in heads:
        num = int(head)
        fout.write(str(idx[num]))
        fout.write(" ")
    fout.write("\n")

           