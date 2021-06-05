#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:14:42 2020

@author: xue
"""
fout = open("z_p_pos.txt","w")
f = open("new_test_tag.txt","r")
findx = open("z_p_idx.txt","r")
idx = []
while True:
    t = findx.readline().split()
    if(len(t) == 0):break
    idx.append(int(t[0]))
i = 0
print(idx)
while True:
    t = f.readline().split()
    if(len(t) == 0):break
   # print(i)
    if i in idx:
       for item in t:
           fout.write(item)
           print(item)
           fout.write(" ")
       fout.write("\n")
    i +=1