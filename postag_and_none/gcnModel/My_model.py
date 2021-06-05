#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:12:37 2019
@author: xue
"""
import copy
import json
import math

import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

MAX_LENGTH= 128
#class GCNClassifier(nn.Module):
#    def __init__(self,num_labels):
#        super().__init__()
#        in_dim = 768
#        self.gcn_model = GCNAbsaModel()
#        self.classifier = nn.Linear(in_dim, num_labels)
#    def forward(self, inputs):
#        outputs = self.gcn_model(inputs)
#        logits = self.classifier(outputs)
#        return logits, outputs

class GCNAbsaModel(nn.Module):
    def __init__(self):
        super().__init__()
        # gcn layer
        self.gcn = GCN()

    def forward(self, head, gcn_input):
   #     print("head"+ str(head.size()))
      #  print(head)
        def inputs_to_tree_reps(head):
            adj = np.zeros((MAX_LENGTH, MAX_LENGTH), dtype=np.float32)

            
            for i in range(0,MAX_LENGTH):
                if head[i] != -1 :
                    # adj[0][i] = 1
                     adj[i][head[i]] = 1
                adj[i][i] = 1;
                   #  adj[i][root] = 1
            #adj = adj * adj
            return adj
        
       # adj = [inputs_to_tree_reps(onehead,typee,loc).reshape(1, MAX_LENGTH, MAX_LENGTH) for onehead in head]
        adj = []  
        for i,onehead in enumerate(head):
            adj.append(inputs_to_tree_reps(onehead).reshape(1, MAX_LENGTH, MAX_LENGTH) )
        adj = np.concatenate(adj, axis=0)
        adj = torch.from_numpy(adj)
        
        adj.cuda()
        h = self.gcn(adj, gcn_input)
        #outputs = h
                # avg pooling asp feature
#        print(h.size())
       # outputs = (h).sum(dim=2)  
       # print(outputs.size())
       # print("outputs"+str(outputs.size()))
        return h

class GCN(nn.Module):
    def __init__(self, num_layers = 2,gcn_dropout = 0.5):
        super(GCN, self).__init__()
        self.gcn_drop = nn.Dropout(gcn_dropout)
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            self.W.append(nn.Linear(768, 768))



    def forward(self, adj, gcn_inputs,layer =2):
        # gcn layer
        #gcn_inputs.cuda()
        denom = adj.sum(2).unsqueeze(2) + 1    # norm
       # print("adj"+ str(adj.size()))
      #  print("gcn_inputs"+ str(gcn_inputs.size()))
        for l in range(layer):
            Ax = torch.bmm(adj.cuda(),gcn_inputs.cuda())
            AxW = self.W[l](Ax)
            AxW = AxW/ denom.cuda()
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < layer - 1 else gAxW
        return gcn_inputs



class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])
    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained("./abc/")
        self.dropout = nn.Dropout(0.1)
        self.classifier_detection = nn.Linear(768, 2)
        self.classifier_sentiment = nn.Linear(768, 4)
        self.embedding = nn.Embedding(5,768)
        #self.embedding.weight.data.copy_(self.load_aspect_embedding_weight())
        self.we_d = nn.Linear(768,768)
        self.we_c = nn.Linear(768,768)
        self.wh_d = nn.Linear(768,768)
        self.wh_c = nn.Linear(768,768)
        self.wa_d = nn.Linear(768,768)
        self.wa_c = nn.Linear(768,768)
        self.w_d = nn.Linear(768,1)
        self.w_c = nn.Linear(768,1)
        self.softmax_d = nn.Softmax(dim=1)
        self.softmax_c = nn.Softmax(dim=1)
        self.gcn = GCNAbsaModel()
      #  self.var = torch.tensor(0.9,dtype=torch.float, device="cuda",requires_grad=True)
    def load_aspect_embedding_weight(self):
        f = open("aspect_embedding/aspect_embedding.txt","r")
        weight = []
        while True:
            line = f.readline()
            if len(line) ==0:break
            item = line.split()
            aspect = []
            for num in item:
                num = float(num)
                aspect.append(num)
            weight.append(aspect)
        myw = torch.tensor(weight,dtype = torch.float)
        return myw
    def forward(self, input_ids, token_type_ids, attention_mask, class_labels, detection_lablels,aspects,all_noun_label,all_sent,head):
        aspect_index =  torch.tensor([3976,2060,2833,25545,2326]);
    #    print(aspects)
        Neg_INF = -1e10
        encode, pooled_output = self.bert(input_ids, attention_mask=attention_mask,token_type_ids= token_type_ids,)
        #print(encode.size())
        
        pooled_output = self.dropout(pooled_output)
        encode = self.dropout(encode)
     #   detection_logits = self.classifier_detection(pooled_output)
       # sentiment_logits = self.classifier_sentiment(pooled_output)
        aspect_embed = self.embedding(aspects)
       # aspect_embed = self.bert.embeddings.word_embeddings(aspect_index[aspects].cuda()).cuda()
        aspect_embed = aspect_embed.unsqueeze(1)
        full_aspect_embed = aspect_embed.expand(-1,128,768)
        
        noun_label = all_noun_label.unsqueeze(-1)
        temp_noun_label = noun_label.repeat(1,1,1)
        noun_label = temp_noun_label * Neg_INF + 1#* self.var
       # detect_encode = encode * noun_label
        #detect_encode = self.dropout(detect_encode)
        
        Md = self.wh_d(encode)*(full_aspect_embed)#+self.we_d(encode)
        attention_d = self.softmax_d(self.w_d(Md) * noun_label)
        temp_encode = encode.permute(0,2,1)
        r_d = torch.bmm(temp_encode,attention_d).squeeze(-1)
        detection_logits = self.classifier_detection(r_d)
        
        my_d = encode * attention_d 
        abc = self.gcn(head,my_d) 
        
        sent_label = all_sent.unsqueeze(-1)
        temp_sent_label = sent_label.repeat(1,1,1)
        un_noun_label =temp_sent_label* Neg_INF + 1
        #un_noun_label =( 1.0 - temp_noun_label) * Neg_INF + 1
        #sent_encode = encode * un_noun_label
        Mc = self.wh_c(encode)*(full_aspect_embed)+self.we_c(Md) + abc
        attention_c = self.softmax_c(self.w_c(Mc) * un_noun_label) #+ attention_d
      #  attention_c = attention_c + attention_d
        temp_encode = encode.permute(0,2,1)
        r_c = torch.bmm(temp_encode,attention_c).squeeze(-1)
        sentiment_logits = self.classifier_sentiment(r_c)
        
      #  print(attention_d.size())
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(detection_logits, detection_lablels)

        loss_fct_2 = CrossEntropyLoss(ignore_index=4)
        loss = loss + loss_fct_2(sentiment_logits,class_labels)
        #print(attention_d.size())
        attention_d = attention_d.squeeze(-1)
#        attention_c = attention_c.squeeze(-1)
#        #print(attention_d.size())
#        #print(attention_d.permute(1,0).size())
#        
       # detection_labels = .
        sizee = full_aspect_embed.size(0)
#      #  print(sizee)
        
        my_detection_labels = detection_lablels.gt(0)
       # print(my_detection_labels)
        my_detection_labels = my_detection_labels.repeat(128,1).permute(1,0)
       # print(my_detection_labels.size())
     #   print(attention_d.size())
        detect_attention_d = torch.masked_select(attention_d,my_detection_labels)
        attention_loss_d = 1 - torch.sum(torch.mul(detect_attention_d,detect_attention_d))/sizee
#        if(attention_loss_d < 0):
#            print(attention_loss_d)
#            print(attention_d)
#        attention_loss_c = 1 - torch.sum(torch.mul(attention_c,attention_c))/sizee
        #print( torch.sum(torch.mul(attention_d,attention_d)))
        loss = loss +attention_loss_d #+attention_loss_c
      #  print(self.var)
      
        

      #  attention_loss_c = 1 - torch.sum(torch.mul(attention_c,attention_c))/sizee
       # loss = loss + attention_loss_c
        return loss, detection_logits,sentiment_logits