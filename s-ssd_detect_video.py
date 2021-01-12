#encoding=utf8

import time
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

cap=cv2.VideoCapture('ceshi.mp4')

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)
        #caffe.set_device(0)
        caffe.set_mode_gpu()
        #caffe.set_mode_cpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self,image,conf_thresh=0.5, topn=5):
        
        image2=image.copy()
        size=image.shape
        height=size[0]
        width=size[1]
        
        #width, height = image.size
        print width, height
        image = cv2.resize(image, (300,300))
        img = np.zeros((1,3,300,300))
        img[0,0,:,:] = image[:,:,0]-104.0
        img[0,1,:,:] = image[:,:,1]-113.0
        img[0,2,:,:] = image[:,:,2]-127.0
        self.net.blobs['data'].data[...] = img
        


        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        #print result
        
        for item in result:
            xmin = int(round(item[0] * width))
            ymin = int(round(item[1] * height))
            xmax = int(round(item[2] * width))
            ymax = int(round(item[3] * height))
            p1 = (xmin, ymin)
            p2 = (xmax, ymax)
            cv2.rectangle(image2, p1, p2, (0,0,255),2)
            p3 = (max(p1[0], 15), max(p1[1]-5, 15))
            title = "%s:%.2f" % (item[-1], item[-2])
            cv2.putText(image2, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
            
           
         
        return result,image2


        

confirm=[7,8,9,10,11]
def main(args):
    '''main '''
    detection = CaffeDetection(args.gpu_id,args.model_def, args.model_weights,args.image_resize, args.labelmap_file)

    k=0
     
    while (True):
        ret,img=cap.read()
        img=cv2.resize(img,(640,480),interpolation=cv2.INTER_CUBIC)
        start_time = time.time()
        result,image2 = detection.detect(img)


###########
        list_con=[]
        if result:
            #print result
            list_con=result[0]
            print list_con
            
            #if list_con[0]==list_con[1] and  list_con[1]==list_con[2] and  list_con[2]==list_con[3] and  list_con[3]==list_con[4] :
               # cv2.putText(image2,'%s' % item[-1],(int(600), int(20),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,155,255),2)
        else:
            list_con=[0,0,0,0,0,0,' ']
            print list_con
        confirm[k]=int(list_con[4])
        k=k+1
        print 'k=%d' % (k)
        if k==4:
            k=0
            print confirm
           
        if confirm[0]==confirm[1] and  confirm[1]==confirm[2] and  confirm[2]==confirm[3] and confirm[0]!=0 and confirm[3]!=0 :
            print 'information'
            #cv2.putText(image2,'%d' % confirm[1],(500, 30),cv2.FONT_HERSHEY_COMPLEX,2,(0, 0, 255),2)
        else:
            pass
            #cv2.putText(image2,'%d' % confirm1[0],(int(600), int(20),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,155,255),2)
##########



        cv2.imshow('img',image2)
        print("FPS: ",1.0/(time.time()-start_time))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='labelmap_voc.prototxt')
    parser.add_argument('--model_def',
                        default='SHN_ssd_33_deploy_depth.prototxt')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--model_weights',
                        default='SN_ssd_33_iter_80000.caffemodel')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
    
