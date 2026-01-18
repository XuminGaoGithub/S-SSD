# S-SSD(A lightweight object detection network for mobile robots by improving the SSD network)

It is a part of work for following paper: 

Gao, X., Jiang, L., Guang, X. and Nie, W., 2020, November. Real-time indoor semantic map construction combined with the lightweight object detection network. In Journal of Physics: Conference Series (Vol. 1651, No. 1, p. 012142). IOP Publishing.



# Demo video:

1)https://www.bilibili.com/video/BV1Lk4y1y7Q7

2)https://www.bilibili.com/video/BV1WZ4y1G7AG


# Requirements
- Cuda 8.0

- Python 2.7

- Caffe 1.0

- Opencv 3.4

- When finish above,it needs install and configure SSD,you can refer to the link (https://blog.csdn.net/a8039974/article/details/79836564)


- It also needs to add the shuffle_channel_layer to Caffe library, you can refer to the link (https://blog.csdn.net/qq_38451119/article/details/82657510) 


# Usage

### Dataset

-Prepare your detection dataset using VOC format.

### Train

-It can refer to the training steps of SSD.

### Predict

- Run `python s-ssd_detect.py` to predict the images.

- Run `python s-ssd_detect_camera.py` to predict by using the camera.

- Run `python s-ssd_detect_video.py` to predict for the videos.


# Abstract

The S-SSD which is a lightweight object detection network based on the SSD network. The innovation of S-SSD is that we designed an efficient and lightweight feature extraction network to replace the VGG network. Compared with the original SSD, the detection speed of S-SSD increased by 2.6 times, while keeping the detection accuracy basically static. It was able to simultaneously meet detection accuracy and real-time performance for the object detection of indoor mobile robots.

1.S-SSD

![Image text](https://github.com/XuminGaoGithub/S-SSD/blob/main/S-SSD.png)

2.The comparison results of different detection networks performance on real scenes

![Image text](https://github.com/XuminGaoGithub/S-SSD/blob/main/1.png)

3.Indoor semantic map construction using mobile robots

![Image text](https://github.com/XuminGaoGithub/S-SSD/blob/main/2.png)


