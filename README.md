CampNet-for-UAV-images-detection
====================================
introduction
-------------------
        CampNet is a object detection model used to detect defective pins in transmission lines. Three improvements   
    have been made based on Faster-RCNN. 
   ***1.Multi-layer feature fusion:*** 
    
        In the UAV inspection images, some critical components in the tower such as bolts often have a small 
    size, and the feature used to distinguish between normal and defective bolts is smaller (nearly 15 pixel). 
    Faster R-CNN only feeds the deep feature layer to region proposal network (RPN), and loses detail information  
    of small-size objects which leads to poor detection. By visualizing the feature map of each layer in the 
    network, we find that the shallow layer helps the network locate the object, while the deep features
    help to classify these objects. Therefore, the deep feature and shallow feature are fused together to by 
    using feature pyramid network (FPN) to identify small-size objects, especially the defective bolts 
    described in this method.
    
   ***2.Context information fusion:*** 
   
        Fuse the object feature information and its contextual information to reduce the false detection in 
    the complex image background. In the inspection images, there are some objects highly like the bolts, 
    and they also would be detected as regions of interest (RoI) by RPN. Because a bolt usually exists on 
    a particular part of a power transmission tower, the surrounding information around the bolt helps to 
    distinguish the defective bolts from similar objects. We tried three different fusion strategies, all 
    of which improve the detection accuracy of defective bolts to varying degrees. This proves that the 
    contextual information helps to reduce the false detection of similar objects indeed. 
    
   ***3.Detector enhancement:*** 
    
        Residual units, which are adopted in ResNet, are added to perform convolution on RoI before they 
    are fed to the detector. From the statistics of the experiments, we found most RoIs come from the 
    shallow feature map, which contains rich detail information but weaker semantic information. The 
    semantic information is usually contained in the deep feature map which is helpful to improve 
    classification accuracy.   
 
Configuration Environment
---------------------
    Ubuntu18.04 + Python 3.5/3.6 + CUDA 10.0 + Tensorflow 1.14.0
    
Installation
--------------------
Clone the repository    
  ```Shell    
  git clone https://github.com/ncepu-epai/CampNet-for-UAV-images-detection.git    
  ```       
Train and Test
--------------------
 Model training and testing refer to https://github.com/yangxue0827/FPN_Tensorflow.

Code of Improvement
---------------------

***1.Multi-layer feature fusion:*** 
The code is located on `./libs/networks/slim_nets/resnet.py(row 109)`

 ***2.Context information fusion:*** 
 
 
 ***3.Detector enhancement:*** 
