import numpy as np
import torch
import torch.random
import json
import random
import os
    
    
class Params():
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
        


class RunningAvg:
    def __init__(self):
        self.metric_sum=0
        self.metric_num=0
    def get_val(self):
        if self.metric_num == 0:
            return 0
        return self.metric_sum/self.metric_num
    def flush(self):
        self.metric_sum=0
        self.metric_num=0
    def append(self, val):
        if isinstance(val, torch.Tensor) and (torch.isinf(val) or torch.isnan(val)):
            return
        self.metric_sum+=val
        self.metric_num+=1

def set_random_seed(seed=5):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
