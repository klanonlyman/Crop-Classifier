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
        CW=img.shape[1]//2
        CH=img.shape[0]//2
        X=sub["target_x"]*-1
        Y=sub["target_y"]*-1
        pointX=CW+X
        pointY=CH+Y
        if pointY-(CH//2)<0:
            starty=0
        else:
            starty=pointY-(CH//2)

        if pointX-(CW//2)<0:
            startx=0
        else:
            startx=pointX-(CW//2)
        if X!=0 or Y!=0:
            img=img[starty:pointY+(CH//2),startx:pointX+(CW//2),:]
        path=path.replace("train","part-split")
        try:
            img = cv2.resize(img, (384,384), interpolation=cv2.INTER_AREA)
            cv2.imwrite(path,img)
                
        except:
            error.append(os.path.join(corr[name],name))
    return error     
trainpath="train"
savepath="part-split"
csvname="tag_locCoor.csv"

mkfolder(trainpath,savepath)
corr=correspond(trainpath)
error=process(trainpath,savepath,csvname,corr)
print(error)