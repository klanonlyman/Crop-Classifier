

import os
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import collections
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x
json_file = open('class_indices.json', "r")
class_indict = json.load(json_file)
floder="" #0.選擇你要ensemble的資料夾
file=os.listdir(floder)
print(file)
weight=[]
img_label={}
for name in file:
    file_name=os.path.join(floder,name)
    df=pd.read_csv(file_name)
    for index in range(0,len(df)):
        sub=df.loc[index]
        img=sub["img"]
        if img not in img_label:
            img_label[img]=np.zeros((33)).astype("float32")
        for i in range(0,33):
            p=sub[str(i)]
            img_label[img][i]+=p
for i in img_label:
    for j in range(0,len(img_label[i])):
        img_label[i][j]=img_label[i][j]/len(file)         
with open(floder+".csv", "a+",newline="") as csvfile:
    writeCsv = csv.writer(csvfile)
    writeCsv.writerow(["filename","label"])
    for img in img_label:
        label=class_indict[str(np.argmax(softmax(img_label[img])))]
        writeCsv.writerow([img,label])