# Crop-Classifier
AI CUP 2022 Crop competition

 <p>
 環境的部分可以參考以下建構:
      https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md
 </p>
<p>
介紹每一個py檔的功能:
  <blockquote>
  <p>
    
  資料前處理的PY檔:
    <p>
      preprocess_for_traning_partsplit、preprocess_for_traning_nonsplit、preprocess_for_traning_allsplit:
      <blockquote>
                依照tag_locCoor.csv檔，將訓練集的數據做切割跟resize，讀取train的資料夾，將分別存成三個處理過的資料all-split、non-split、part-split(這些為等等model的訓練集資料)。</blockquote>
       </br>
      preprocess_for_testing_partsplit、preprocess_for_testing_nonsplit、preprocess_for_testing_allsplit:<blockquote>
                依照tag_loccoor_public.csv檔，將測試集做跟訓練集的數據一樣的前處理，讀取public or private的資料夾，將分別存成三個處理過的資料part_public、non_public、all_public(這些為等等model的預測集資料)，如果要處理ptivate的資料集則需要修改程式碼中#0.讀取路徑、#1.保存路徑、#2.切割EXCEL。</blockquote>
      </p>
  </p>  
  <p>
  model訓練後的PY檔:
  </br> </br>
      predict_private_V1、predict_public_V1、predict_private_V2、predict_public_V2:
     
  <blockquote>
                將手動選擇你需要用那些weights做model soup，最後預測完的結果會存成CSV檔，每個CSV檔代表一個model soup預測的結果(還沒有softmax)，需修改程式碼中的#0.選擇epoch、#1.選擇epoch、#2.選擇要預測的資料集、#3.選擇權重資料集，去挑選最好的結果。(note:predict_public_V1、predict_public_V2的CSV檔會存在result_public的資料夾內，反之private的PY檔會存result_private的資料夾)     
    </blockquote>
    </br>
      ensemble:
 <blockquote>
                將result_private or result_public的資料夾內的預測的結果做ensemble，裡面會把結果作相加，然後在平均，之後再做softmax來當最後的分類結果。(note:程式碼內需修改#0.選擇你要ensemble的資料夾，最後會得到兩分類結果的CSV檔(result_public.csv and result_private.csv))
    </blockquote>
  </p> 
  <p>
  model部分的PY檔: </br>
 <blockquote>
      swin_transformer_v1、swin_transformer_v2 : 我使用的兩個model的架構
  </blockquote>
 <blockquote>
      train_V1、train_V2 :分別訓練對應的model跟資料集， 需要修改#0.定義weight保存路徑、#1.定義LR、#2.定義訓練的資料集、#3.定義預訓練的weight
 </blockquote>
 </p>
 </blockquote>
 </p>
 

 <p>
  
 程式執行前準備:</br>
 <blockquote>
      train資料夾:放訓練集資料</br>
      public資料夾:放測試集資料</br>
      private資料夾:放測試集資料  </br>       
      result_private和result_public資料夾:放預測結果(一開始裡面要是空的)</br>
      tag_locCoor.csv、tag_loccoor_private.csv、tag_loccoor_public.csv:資料準心切割依據</br>
      </blockquote>
  </p>
   <p>
 程式執行順序:</br>
 <blockquote>
      traing dataset prepare:</br>
      <blockquote>
        1.preprocess_for_traning_partsplit.py</br>
        2.preprocess_for_traning_nonsplit.py</br>
        3.preprocess_for_traning_allsplit.py</br>
        </blockquote>
      testing dataset prepare(自行修改#標住的地方，為了public和private dataset):</br>
      <blockquote>
        4.preprocess_for_testing_partsplit.py</br>
        5.preprocess_for_testing_nonsplit.py</br>
        6.preprocess_for_testing_allsplit.py</br>
        </blockquote>
      NOTE:此階段會得到9個資料夾</br>
      model train:</br>
      <blockquote>
         1.train_V1.py (訓練全部沒切的資料集，拿取官方pretrain weight)，將會得到一個資料夾(全部epoch的weight)</br>
         2.執行 soup_V1.py 得到平均的權重W1(自行選擇需要拿那些epoch)#0.選擇epoch、#1.選擇epoch、 #2.選擇weights位址、#3.保存model soup後的名稱.pth</br>
         3.train_V1.py (訓練全部部分切的資料集，拿W1當pretrain weight)，將會得到一個資料夾(全部epoch的weight)</br>
         4.執行 soup_V1.py 得到平均的權重W2(自行選擇需要拿那些epoch)#0.選擇epoch、#1.選擇epoch、 #2.選擇weights位址、#3.保存model soup後的名稱.pth</br>
         5.train_V1.py (訓練全部全切的資料集，拿W2當pretrain weight)，將會得到一個資料夾(全部epoch的weight)</br>
         </blockquote>
         以上步驟自行將V1換成V2即可</br>
      model predict:</br>
      <blockquote>
         1.執行predict_public_V1.py (修改使用不同資料集的model搭配不同的資料集，以及model soup拿取不同的epoch)</br>
         2.執行predict_public_V2.py (修改使用不同資料集的model搭配不同的資料集，以及model soup拿取不同的epoch)</br>
         3.執行ensemble.py (需修改對映的path) 這時會產生一個result_public.CSV檔</br>
         </blockquote>
         以上將public改為private即可，note:個人在public和private在預測時都是使用相同的model soup的epoch。</br>
  </blockquote>
  </p>
