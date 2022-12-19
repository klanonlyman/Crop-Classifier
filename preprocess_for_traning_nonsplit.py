import os
import pandas as pd
import cv2
import numpy as np


def mkfolder(trainpath,savepath):
    if not(os.path.exists(savepath)):
        os.mkdir(savepath)
    classes=os.listdir(trainpath)
    for cla in classes:
        path=os.path.join(savepath,cla)
        if not(os.path.exists(path)):
            os.mkdir(path)
def correspond(trainpath):
    corr={}
    classes=os.listdir(trainpath)
    for cla in classes:
        path=os.path.join(trainpath,cla)
        imgs_path=os.listdir(path)
        for imgname in imgs_path:
            corr[imgname]=path
    return corr
def process(trainpath,savepath,csvname,corr):
    df=pd.read_csv(csvname,encoding="big5")
    error=[]
    for i in range(0,len(df)):
        sub=df.loc[i]
        name=sub["Img"]   
        path=os.path.join(corr[name],name)
        img=cv2.imread(path)
        path=path.replace("train",savepath)
        try:
            img = cv2.resize(img, (384,384), interpolation=cv2.INTER_AREA)
            cv2.imwrite(path,img)
        except:
            error.append(os.path.join(corr[name],name))
    return error     
trainpath="train"
savepath="non-split"
csvname="tag_locCoor.csv"

mkfolder(trainpath,savepath)
corr=correspond(trainpath)
error=process(trainpath,savepath,csvname,corr)
print(error)