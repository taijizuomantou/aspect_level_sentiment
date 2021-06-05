#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 22:56:52 2021

@author: xue
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:59:58 2021

@author: xue
"""

import torch
model = torch.load("model_data/latestsentimodel4")
torch.save(model.state_dict(), "model_data/latestsenti")