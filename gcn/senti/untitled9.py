#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:59:58 2021

@author: xue
"""

import torch
model = torch.load("model_data/attention_add_model1")
torch.save(model.state_dict(), "model_data/senti")