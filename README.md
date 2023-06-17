# Sky-Segmentation-and-Post-processing
This is a C++ implementation from this paper https://arxiv.org/abs/2006.10172 that published on 2020, the repo is for sky mask post-processing. but I didn't implemente the "Density Estimation" mentioned in the paper. 

About Sky segmentation, I trained the sky-segmentation model by U-2-Net, the result looks good. please refer to https://github.com/xuebinqin/U-2-Net about training detail

Dependency：OpenCV, ncnn

seg_demo.cpp is for sky-seg and input is image 

mask_refine.cpp is for mask post-process to refine the mask. inputs are image and the mask inferenced by model.

The Sky-mask Post-Processing show a good performence in the scene of tree as below. it retain much more details.In addition, the post-process is only for sky-mask.perhaps it won't get the same good performance when you apply it on other class segmentation.

**2021/12/29 Update: upload code interenced by onnxruntime, you need to install the package by pip install onnxruntime**

onnx model(167M) baiduyun：https://pan.baidu.com/s/1bE38w422STSwuJwjPpRIMw      code：4tmm

**2021/10/13 Update**

**Upload a small sky-seg model of 2Mb（traind by u2netp） for demo（We couldn't public the high-precision model because it used in our product）**

**Upload a sky-seg demo cpp inferenced by ncnn**

![vis2](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing/blob/main/vis2.png)

![vis1](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing/blob/main/vis1.png)

**but it also has some defect：in the scene of building, some detail of building will be considered as sky by mistake**

![vis3](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing/blob/main/vis3.png)

**For some special textured clouds, The algorithm has some flaws as below**

![vis4](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing/blob/main/vis4.png)

**Next TODO: the U-2-Net couldn't run in real-time in mobile device(about 300ms in Snapdragon 888). even though u2netp size is much smaller than u2net, but the interence speed doesn't improve obviously. I plan to train a real-time model by normanl unet so that it could run in real-time in mobile device.**

![img](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing/blob/main/output.gif)

![fun1](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing/blob/main/fun1.png)
