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
        self.posembedding = nn.Embedding(20,768)
        self.p = nn.Linear(50,768)
        #self.embedding.weight.data.copy_(self.load_aspect_embedding_weight())
        self.we_d = nn.Linear(768,768)
        self.we_c = nn.Linear(768,768)
        self.wh_d = nn.Linear(768,768)
        self.wh_c = nn.Linear(768,768)
        self.wa_d = nn.Linear(768,768)
        self.wa_c = nn.Linear(768,768)
        self.wfa_d = nn.Linear(768,768)
        self.wfa_c = nn.Linear(768,768)
        
        self.wete_d = nn.Linear(768,768)
        self.wete_c = nn.Linear(768,768)
        
        self.w_d = nn.Linear(768,1)
        self.w_c = nn.Linear(768,1)
        self.softmax_d = nn.Softmax(dim=1)
        self.softmax_c = nn.Softmax(dim=1)
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
    def forward(self, input_ids, token_type_ids, attention_mask, class_labels, detection_lablels,aspects,all_noun_label,all_sent):
        encode, pooled_output = self.bert(input_ids, attention_mask=attention_mask,token_type_ids= token_type_ids,)
        #print(encode.size())
        pooled_output = self.dropout(pooled_output)
        encode = self.dropout(encode)
        
        

        aspect_embed = self.embedding(aspects)
        aspect_embed = aspect_embed.unsqueeze(1)
        full_aspect_embed = aspect_embed.expand(-1,128,768)
        
        noun_label = all_noun_label.unsqueeze(-1)
        noun_label = noun_label.repeat(1,1,768)
        noun_label = noun_label #* self.var
        detect_encode = encode * noun_label

     
        Md = self.wh_d(detect_encode)+self.wa_d(full_aspect_embed)
        attention_d = self.softmax_d(self.w_d(Md))
        temp_encode = encode.permute(0,2,1)
        r_d = torch.bmm(temp_encode,attention_d).squeeze(-1)

        
        detection_logits = self.classifier_detection(r_d)
        

        
        Mc = self.wh_c(encode)+self.wa_c(full_aspect_embed)
        attention_c = self.softmax_c(self.w_c(Mc)) #+ attention_d
        attention_c = attention_c + attention_d
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

       # detection_labels = .
        sizee = full_aspect_embed.size(0)
#      #  print(sizee)
        
        my_detection_labels = detection_lablels.gt(0)

        my_detection_labels = my_detection_labels.repeat(128,1).permute(1,0)

        detect_attention_d = torch.masked_select(attention_d,my_detection_labels)
        attention_loss_d = 1 - torch.sum(torch.mul(detect_attention_d,detect_attention_d))/sizee

        loss = loss +attention_loss_d #+attention_loss_c
      #  print(self.var)
        return loss, detection_logits,sentiment_logits