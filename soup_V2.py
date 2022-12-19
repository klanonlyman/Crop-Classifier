

import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from swin_transformer_v2 import swin_large_patch4_window7_384_in1k as create_model
import numpy as np
import csv
import collections
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
def test(start,end,weightpath):
    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models=[]
    for item in range(start,end):
        model = create_model(num_classes=num_classes).to("cpu")
        model_name = 'model-%s'%item
        model_weight_path = weightpath+"\\"+model_name+".pth"
        print(model_weight_path)
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        models.append(model)
        
    fl_model=create_model(num_classes=num_classes).to(device)
    worker_state_dict = [x.state_dict() for x in models]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    
    for key in weight_keys:
        key_sum = 0
        for i in range(len(models)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(models)
    
    return fed_state_dict
    
    
if __name__ == '__main__':
    start= #0.選擇epoch
    end=  #1.選擇epoch
    weightpath="" #2.選擇weights位址
    savename="" #3.保存model soup後的名稱.pth
    newweight=test(start,end,weightpath)
    model = create_model(num_classes=33)
    model.load_state_dict(newweight, strict=False)
    torch.save(model.state_dict(), savename)