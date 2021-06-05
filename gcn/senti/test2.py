#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 22:46:03 2020

@author: xue
"""

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
abc = torch.tensor([0, 2, 2, 1, 3, 4, 1, 0, 0, 4, 3, 2, 0, 4, 4, 4], device='cuda:0')
aspect_index =  torch.tensor([3976,2060,2833,25545,2326]);
ans = aspect_index[abc]
print(ans)
bert = BertModel.from_pretrained("./abc/")
print(bert.embeddings.word_embeddings(ans))