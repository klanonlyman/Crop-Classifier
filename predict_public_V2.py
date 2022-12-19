




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
def test(start,end,datapath,weightpath,savename,ensemble,num_classes):
    json_file = open('class_indices.json', "r")
    class_indict = json.load(json_file)
    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
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
    #### update fed weights to fl model
    fl_model.load_state_dict(fed_state_dict)       
    fl_model.eval()
    fl_model.cuda()
    
    number=0
    file = os.listdir(datapath)
    
    
    for img_name in file:
        img_path = os.path.join(datapath,img_name)
        img = Image.open(img_path)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = torch.squeeze(fl_model(img.to(device))).cpu()
            
        ensemble_output = output.numpy()
        temp={"img":img_name}
        for idx in range(0,num_classes):
            temp[str(idx)]=ensemble_output[idx]
        ensemble=ensemble.append(temp,ignore_index=True)
        number=number+1
        if number%100==1:
            print("img:",number)
    ensemble.to_csv("./result_public/"+str(start)+"-"+str(end)+savename+".csv",index=False)
if __name__ == '__main__':
    column=['img']
    num_classes=33
    for i in range(0,num_classes):
        column.append(str(i))
        ensemble = pd.DataFrame(columns=column)
    start= #0.選擇epoch
    end= #1.選擇epoch
    datapath="" #2.選擇要預測的資料集
    weightpath="" #3.選擇權重資料集
    savename=weightpath
      
    test(start,end,datapath,weightpath,savename,ensemble,num_classes)
    
  


