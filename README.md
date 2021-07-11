# AGM

**\emph{The main contributors} of this paper are Junjie Huang(JoJoliking) and Zhibo Zou(460115062ian)**

**Apperance Guidance Attention for Multi-Object Tracking:**

![Image text](https://github.com/460115062ian/picture/blob/main/3.png)

## Abstract
Appearance information is one of the most important matching indicators for multi-object data association. In tracking by detection model, appearance information and detection information are usually integrated in the same sub-network for learning and output. This will cause the appearance embedding vector to be coupled to the network inference method during the learning process. As a result, the appearance embedding vector contains too much background information and affects the accuracy of data association. Based on the non-end-to-end tracking model, we design an appearance guidance attention module on the appearance extraction branch. This module can effectively strengthen the network’s learning of the object visual appearance features and reduce the attention to the learning of background features. Finally, we use the appearance embedding vector that decoupled from the inference method as the input of the back-end tracker and perform data association. Our method is tested on theMOT16 andMOT17 datasets. Experiments show that our method provides more high-quality appearance representation information for the back-end tracker and the tracking performance on the two datasets is better than other comparison models. At the same time, our model can reach 24.9FPS on a single 1080Ti.
## Notices
**The code is an unorganized version.**
## Tracking Performance
**Results on MOTchallenge test set**
|Dataset|MOTA|IDF1|MOTP|MT|ML|Recall|ID.Sw|
|--------|-------|-----|---|---|--|---|---|
|MOT16|68.3|68.3|80.0|40.4%|13.8%|77.1|1005|
|MOT17|68.1|67.7|80.1|39.0%|14.9%|75.6|3195|

All of the results are obtained on the [MOT challenge](https://motchallenge.net/) evaluation server under the “private detector” protocol. We rank first among all the trackers on  MOT16 and MOT17. 

<img src="https://github.com/460115062ian/picture/blob/main/1.gif" width="320" height="180"/><img src="https://github.com/460115062ian/picture/blob/main/22.gif" width="320" height="180"/>

## Requirements 
Python 3.8 or later with all [requirments.txt](https://github.com/JoJoliking/AGM/blob/main/requirements.txt) dependencies installed, including `   torch>=1.7`. To install run:
` $ pip install -r requirements.txt`

 ## Training
* Download the training datasets
* ` $ python train.py --data ./dataset/MOT.yaml --hyp ./dataset/hyp.scractch.yaml --batch-size 16` 
## Acknowledgement
A large part of the code is borrowed from [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT) and [ultralytics/yolov5](https://github.com/ultralytics/yolov5). Thanks for their wonderful works.

