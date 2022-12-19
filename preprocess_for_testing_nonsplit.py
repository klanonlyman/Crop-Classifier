import os
import pandas as pd
import cv2
import numpy as np

def process(trainpath,savepath,csvname,size):
    df=pd.read_csv(csvname,encoding="big5")
    error=[]
    for i in range(0,len(df)):
        sub=df.loc[i]
        name=sub["Img"]
        path=os.path.join(trainpath,name)
        img=cv2.imread(path)
        try:
            img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(savepath,name),img)        
        except:
            error.append(os.path.join(trainpath,name))
    return error
size=384
trainpath="public"#0.讀取路徑
savepath="non_public" #1.保存路徑
os.mkdir(savepath)
csvname="tag_loccoor_public.csv"  #2.切割EXCEL

error=process(trainpath,savepath,csvname,size)
print(error)