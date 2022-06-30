
# Object Detection on COCO Dataset

Ashwani Rajan, Chahak Sethi

## Goal
The goal of this project is to perform object detection on Common Objects in Context (COCO) Dataset. We aim to get handa-on experience working with the state-of-the-art object detection models by using the pretrained models for inference and by fine-tuning the model for a custom dataset of our interest. We aim to explore two architectures: Yolo-v5 and Faster-RCNN models. Finally, we aim to compare the model performance and training speed of these architectures on the Coco Dataset.

## About the Dataset
[Common Objects in Context (COCO)](https://cocodataset.org/#home) is a large-scale open source dataset for a number of computer vision tasks. It is the benchmark dataset for object detection tasks. It contains arround 118,000 training images and 5000 validation images, along with 80 object categories. For this project, we use a subset of the COCO dataset to perform training and inference for the two models that we use. For Yolov5, we use all the object categories. However, for Faster-RCNN, due to high training time, we have taken only four land-vehicle classes: "car", "bus", "truck" and "bicycle".

## Models and Inference

<img style="float: right; size: 100px" src="./Images/faster-RCNN.png" width="300" height="200">

For this project, we use the two state-of-the-art models for object detection, Yolo-v5 and Faster-RCNN. We started with the [**Faster RCNN model**](https://arxiv.org/abs/1506.01497) for object detection. We used the Faster RCNN model from torchvision with **ResNet-50-FPN backbone** with pretrained weights trained on Imagenet. We directly used the **train_one_epoch** and **evaluate** functions from torchvision's [github repository](https://github.com/pytorch/vision.git) to fine tune the model. 

To use [**Yolo-v5**](https://docs.ultralytics.com/) for object detection.

<Yolo>

## Results

**Faster RCNN model** : Using the pre-trained ResNet-50-FPN network and training for 5 epochs with a constant learning rate of 0.005, we see a constant decline in the loss values. The Average Precision at IOU 50:95 is observed around 0.32 after 5 epochs. We further evaluated the model's performance through f1-score, precision and recall and the confusion matrix and we found these observations:

| metrics | Confusion Matrix |
| ------- | ---------------- | 
| <img src="./Images/metrics.jpg" width="300" height="200"> | <img src="./Images/Cmat2.png" width="300" height="200"> |

Here are some examples of the results:
|  |  |
|----- | ----- |
|<img src="./Images/ex1.png"> | <img src="./Images/ex2.png">|

<Yolo>


## Installations

To use the notebook, you would need to install fiftyone library and latest version of torchvision. Yoo would also need to clone the pytroch's vision repository and copy the required files (engine.py, transforms.py, utils.py) from references folder to your working directory. Once all the dependencies are installed, you should be able to replicate the results using the **Faster-RCNN-COCO.ipynb** notebook. 

```console 
!pip install fiftyone
!pip install torch torchvision
!pip install opencv-python-headless==4.5.4.60 # needed only if there's an open-cv related error 

git clone https://github.com/pytorch/vision.git
!git clone https://github.com/ultralytics/yolov5 
!cd yolov5
!pip install -qr requirements.txt
```
