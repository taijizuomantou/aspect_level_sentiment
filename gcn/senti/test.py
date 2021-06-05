#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:07:08 2019

@author: xue
"""
import torch.nn.functional as F
import torch
inf =float("-inf")
k = torch.tensor([[inf,0,inf,0],[inf,0,0,0]],dtype =float)


x = torch.tensor([[1,2,3,4],[4,5,1,2]],dtype =float)
print(x + k)
print(F.softmax(x + k,dim = 0))
print(F.softmax(x + k,dim = 1))
#import numpy as np
#import torch
#abc = np.random.randn(4,300)
#torch_abc = torch.from_numpy(abc)
#torch_abc = torch_abc.unsqueeze(-1)
#print(torch_abc.size())
#from My_model import BertForSequenceClassification
#import torch
#label_id = [0,1,2,3,4,4,4,1]
#label_exist_id = []
#    #label_classify_id = []
#for label in label_id:
#   if label == 4:
#        label_exist_id.append(0)
#   else:
#        label_exist_id.append(1)
#label_exist_id = torch.tensor(label_exist_id,dtype = torch.long)
#print(label_exist_id)
#print(label_exist_id.unsqueeze(1))
#aspect_map = {}
#aspect_list = ['price','anecdotes','food','ambience','service']
#for (i, label) in enumerate(aspect_list):
#    aspect_map[label] = i
#print(aspect_map)
#import torch.nn as nn
#embedding = nn.Embedding(5,3)
#for i in range(0,5):
#    print(embedding(torch.tensor(i,dtype=torch.long)))
#
#print(torch_abc.size())
#expand = torch_abc.expand(-1,512,300)
#print(expand.size())
#aa = np.random.randn(4,512,768)
#aa = torch.from_numpy(aa)
#answer = torch.cat((aa,expand),dim = -1)
#print(answer.size())
#aa = np.random.randn(4,512,1)
#aa = torch.from_numpy(aa)
#aa = aa.squeeze(-1)
#print(aa.size())
#a1 = [[2],[4]]
#
#a2 = [[2,3,4],[4,5,6]]
#a1 = torch.tensor(a1,dtype = torch.long)
#a1 = a1.expand(2,3)
#a2 = torch.tensor(a2,dtype = torch.long)
#print(a1.size())
#print(a2.size())
#print(a1)
#ans = torch.mul(a1,a2)
#print(ans)
#abc = np.random.randn(4,300,1)
#torch_abc = torch.from_numpy(abc)
#torch_abc = torch_abc.squeeze(-1)
#print(torch_abc.size())
#def load_aspect_embedding_weight():
#    f = open("aspect_embedding/aspect_embedding.txt","r")
#    weight = []
#    while True:
#        line = f.readline()
#        if len(line) ==0:break
#        item = line.split()
#        aspect = []
#        for num in item:
#            num = float(num)
#            aspect.append(num)
#        weight.append(aspect)
#    myw = torch.tensor(weight,dtype = torch.float)
#    return myw
##weight = load_aspect_embedding_weight()
##model = BertForSequenceClassification( 5)
##no_decay = ['bias', 'gamma', 'beta']
##print(model.named_parameters() )
##for n, p in model.named_parameters() :
##  #  print(n)
##  #pass
##    if any(nd in n for nd in no_decay):
##        print(n)
##optimizer_parameters = [
##     #{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
##     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
##     ]
##i = 0
##f = open("abc2.txt","w")
##f.write(str(optimizer_parameters))
#       # import torch
#x = torch.tensor([[0.3,0.7],[0.2,0.8]])
#t = torch.tensor([[0.3,0.7],[0.9,0.1]])
#print(torch.mul(x,x))
#y = torch.sum(torch.mul(x,x)/2 ) #x.mm(x)
#print(y)
#weight_c=torch.rand(1, dtype=torch.float)
#print(weight_c)
#print(torch.log(weight_c))
#weight_c = torch.rand(1,dtype=torch.float,requires_grad= True).cuda()
#print(weight_c.is_leaf)
#model = torch.load("model_data/attention_add_model6")
#print(model.weight_c)
##print(model.weight_d)
#from processor import (Semeval_NLI_B_Processor, Semeval_NLI_M_Processor,
#                       Semeval_QA_B_Processor, Semeval_QA_M_Processor,
#                       Semeval_single_Processor, Sentihood_NLI_B_Processor,
#                       Sentihood_NLI_M_Processor, Sentihood_QA_B_Processor,
#                       Sentihood_QA_M_Processor, Sentihood_single_Processor)
#processors = {
#        "sentihood_single":Sentihood_single_Processor,
#        "sentihood_NLI_M":Sentihood_NLI_M_Processor,
#        "sentihood_QA_M":Sentihood_QA_M_Processor,
#        "sentihood_NLI_B":Sentihood_NLI_B_Processor,
#        "sentihood_QA_B":Sentihood_QA_B_Processor,
#        "semeval_single":Semeval_single_Processor,
#        "semeval_NLI_M":Semeval_NLI_M_Processor,
#        "semeval_QA_M":Semeval_QA_M_Processor,
#        "semeval_NLI_B":Semeval_NLI_B_Processor,
#        "semeval_QA_B":Semeval_QA_B_Processor,
#    }
#
#processor = processors["sentihood_NLI_M"]()
#train_examples = None
##num_train_steps = None
#train_examples = processor.get_train_examples("data/sentihood/bert-pair/")
#train_examples = processor.get_test_examples("data/sentihood/bert-pair/")
#fin = open("data/sentihood/bert-pair/test_NLI_M.tsv","r")
#for line in fin:
#    item = line.split("\t")
##    print(item[1])
#f = open("train_pos_tag.txt","r")
#f_out = open("pos.txt","w")
#pos_dict = {}
#for line in f:
#    my = f.readline()
#    my_list = my.split()
#    for item in my_list:
#        #print(item)
#        if item not in pos_dict:
#            pos_dict[item] = 0
#        pos_dict[item] += 1
#count = 1
#for item in pos_dict:
#    item = "'"+item+"'"
#    f_out.write(item)
#    f_out.write(':'+str(count))
#    f_out.write(",")
#    count += 1
#   # WP$
#pos_text_to_label = {'CC':1,'DT':2,'NN':3,'VBD':4,'RB':5,'JJ':6,'TO':7,'PRP':8,'.':9,'VB':10,',':11,'WDT':12,'IN':13,"''":14,'NNS':15,'VBP':16,'RP':17,'PDT':18,'VBZ':19,'MD':20,'WRB':21,'WP':22,'POS':23,'PRP$':24,'(':25,')':26,'SYM':27,':':28,'VBG':29,'VBN':30,'CD':31,'JJS':32,'JJR':33,'EX':34,'RBR':35,'$':36,'RBS':37,'UH':38,'NNP':39,'FW':40,'#':41,'LS':42,'WP$':43}
#print(pos_text_to_label['CC'])
--task_name semeval_NLI_M \
--data_dir data/semeval2014/bert-pair/ \
--vocab_file uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
--eval_test \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 16 \
--learning_rate 3e-5 \
--num_train_epochs 6.0 \
--output_dir results/semeval2014/NLI_M \
--seed 42
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.1131],
        [0.0000],
        [0.1311],
        [0.1214],
        [0.1387],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.1532],
        [0.0000],
        [0.0000],
        [0.1182],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.1011],
        [0.0000],
        [0.0000],
        [0.1232],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000],
        [0.0000]],