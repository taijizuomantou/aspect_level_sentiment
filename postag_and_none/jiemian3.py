#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:15:50 2021

@author: xue
"""

import PySimpleGUI as sg
import os
import tokenization
import en_core_web_sm
from spacy import displacy
from showtest import semwithpos,semwithnone,sentiwithpos,sentiwithnone,semwithgcn,sentiwithgcn
nlp = en_core_web_sm.load()
sg.theme('LightGrey1')   # Add a touch of color
# All the stuff inside your window.
layout0 = [[sg.Button("选择数据库")],
            [sg.InputCombo(['SemEval', 'Sentihood'],key="dataset",default_value="Sentihood")],
            [sg.Button('确定')]]

layout_d = [[sg.FileBrowse('打开图片',key='filebrowser',target='image_shape'),sg.InputText('',key='image_shape',disabled=True)]]
# Create the Window
window = sg.Window('Window Title', layout0,font=("黑体", 15))
# Event Loop to process "events" and get the "values" of the inputs
folder = ""
win2_active = False
Dataset = ""

    
while True:
    event, values = window.read(timeout=100)
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    if event == "选择数据库":
        folder = ""
        folder = sg.PopupGetFolder('选择数据库文件夹', default_path='',font=("黑体", 15))
    if event == "确定":
        #folder = "/home/xue/下载/aaaaa"#
        #values["dataset"] = "SemEval"#
        if folder == "" or values["dataset"] == "":
            sg.popup("请选择文件夹和数据库")
            
        else:
            name = os.listdir(folder)
            DataSet = values["dataset"] 
            win2_active = True
            aspectData = []
            if values["dataset"] == "SemEval":
                aspectData = ["food","ambience","service","price","anecdotes"]
            else:
                aspectData = ['location - 1 - general','location - 1 - price','location - 1 - safety','location - 1 - transit location','location - 2 - general','location - 2 - price','location - 2 - safety','location - 2 - transit location']
            layout2 = [  [sg.Text("文档"),sg.InputCombo(name,auto_size_text=True,key="files"),sg.Text("方面"),sg.InputCombo(aspectData,auto_size_text=True,key="aspect",default_value=aspectData[0]),sg.Button('确定')],
            [sg.Multiline(size=[100,5],key="doc")],
            [sg.Multiline(size=[100,10],key="pos"),sg.Button("词性分析")],
            [sg.Multiline(size=[100,10],key="depend"),sg.Button("依存关系分析")],
            [sg.Button('WithNone'),sg.Text(key = "noneans",size=[100,1],auto_size_text= True)],
            [sg.Button('WithPos'),sg.Text(key = "posans",size=[100,1],auto_size_text= True)],
            [sg.Button('WithGCN'),sg.Text(key = "gcnans",size=[100,1],auto_size_text= True)]]
              
            window2 = sg.Window('Window 2', layout2,font=("黑体", 15))
            
    if win2_active:
        events2, values2 = window2.Read(timeout=100)
        if events2 is None or events2 == 'Exit':
          win2_active = False
          window2.close()
        
        if events2 == "确定":
           
           # print(folder)
          #  print(values2["files"])
            if folder != ""  and values2["files"] != "":
                path = folder +"/"+ values2["files"]
               # print(path)
                #sg.FileBrowse('打开图片',key='filebrowser',target='image_shape')
                f = open(path, 'r')
                ans = f.read()
                window2.Element("doc").Update(ans)
        if events2 == "词性分析":
            
                tokenizer = tokenization.FullTokenizer(
        vocab_file="uncased_L-12_H-768_A-12/vocab.txt", do_lower_case=True)
                word,tag=tokenizer.tokenize(ans)
                showtag = ""
                for i in range (0,len(word)):
                    temp = "("+word[i]+","+tag[i]+")"
                    showtag = showtag +temp
                window2.Element("pos").Update(showtag)
        if events2 == "依存关系分析":
                key = len(ans)
                ans = ans[0:key-1]
                parse = nlp(ans)
                showparse = ""
                for item in parse:
                     showparse += ("("+str(item) + "," + str(item.head)+")")
                #displacy.serve(parse, style='dep') // zese 
                window2.Element("depend").Update(showparse) 
        #if events2 == "WithNone":print("ceshi")  filter ans and aspect 
        if DataSet == "SemEval":
            if events2 == "WithNone":
                print(ans)
                aspect = values2["aspect"]
                exist,sent = semwithnone.main(ans,aspect)
                showans = aspect +" "
                if exist == 0:showans += "not exist"
                else:
                    showans +="exist and sentiment polarity is "
                    sentchoice = ['positive', 'neutral', 'negative', 'conflict']
                    showans += sentchoice[sent]
                
                print(showans)
    
                window2.Element("noneans").Update(value=showans)     #do not use same name
            if events2 == "WithPos":
                print(ans)
                aspect = values2["aspect"]
                exist,sent = semwithpos.main(ans,aspect)
                showans =aspect +" "
                if exist == 0:showans += "not exist"
                else:
                    showans +="exist and sentiment polarity is "
                    sentchoice = ['positive', 'neutral', 'negative', 'conflict']
                    showans += sentchoice[sent]
                
                print(showans)
                window2.Element("posans").Update(value=showans)     #do not use same name
            if events2 == "WithGCN":
                print(ans)
                aspect = values2["aspect"]
                exist,sent = semwithgcn.main(ans,aspect)
                showans =aspect +" "
                if exist == 0:showans += "not exist"
                else:
                    showans +="exist and sentiment polarity is "
                    sentchoice = ['positive', 'neutral', 'negative', 'conflict']
                    showans += sentchoice[sent]
                
                print(showans)
                window2.Element("gcnans").Update(value=showans)     #do not use same name
        else:
            if events2 == "WithNone":
                print(ans)
                aspect = values2["aspect"]
                print(aspect)
                exist,sent = sentiwithnone.main(ans,aspect)
                showans = aspect +" "
                if exist == 0:showans += "not exist"
                else:
                    showans +="exist and sentiment polarity is "
                    sentchoice = ['positive', 'negative']
                    showans += sentchoice[sent]
                
                print(showans)
    
                window2.Element("noneans").Update(value=showans)     #do not use same name
            if events2 == "WithPos":
                print(ans)
                aspect = values2["aspect"]
                exist,sent = sentiwithpos.main(ans,aspect)
                showans =aspect +" "
                if exist == 0:showans += "not exist"
                else:
                    showans +="exist and sentiment polarity is "
                    sentchoice = ['positive', 'negative']
                    showans += sentchoice[sent]
                
                print(showans)
                window2.Element("posans").Update(value=showans)     #do not use same name                
               # print("zoudolae")
            if events2 == "WithGCN":
                print(ans)
                aspect = values2["aspect"]
                exist,sent = sentiwithgcn.main(ans,aspect)
                showans =aspect +" "
                if exist == 0:showans += "not exist"
                else:
                    showans +="exist and sentiment polarity is "
                    sentchoice = ['positive', 'negative']
                    showans += sentchoice[sent]
                
                print(showans)
                window2.Element("gcnans").Update(value=showans)     #do not use same name
window.close()

#cixing
#gcn chc chaolianjiechakantupian