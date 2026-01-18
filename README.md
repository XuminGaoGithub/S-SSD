# S-SSD(A lightweight object detection network for mobile robots by improving the SSD network)

Authur：Xumin Gao, et al. (Institute of Robotics and Intelligent Systems, Wuhan University of Science and Technology, China)

It is a part of work for following paper: 

Xumin Gao, et al. Real-time Indoor Semantic Map Construction Combined with The Lightweight Object Detection Network. The 2020 2nd International Conference on Artificial Intelligence Technologies and Applications (ICAITA 2020). Dalian.



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


