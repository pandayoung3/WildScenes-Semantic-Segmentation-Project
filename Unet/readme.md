
# README for UNet and UNet-ASPP on WildScenes Dataset

## Introduction

This project utilizes the UNet and UNet-ASPP models for semantic segmentation on the 2D portion of the **WildScenes** dataset. WildScenes provides a robust benchmark for testing segmentation models in large-scale natural environments, making it ideal for evaluating the performance of these models in complex and diverse scenarios.

## Project Overview

In natural environments, accurately segmenting images is essential for applications such as autonomous navigation, environmental monitoring, and wildlife conservation. This project leverages the UNet and UNet-ASPP models to address the challenges posed by the WildScenes dataset, aiming to achieve precise segmentation of natural scenes, including various terrains, vegetation, and objects.

### Goals

- **Segment Complex Natural Scenes:** Accurately identify and segment natural elements within diverse environments.
- **Evaluate Model Performance:** Compare the effectiveness of UNet and UNet-ASPP on the WildScenes dataset.
- **Improve Segmentation Techniques:** Enhance current segmentation methods for real-world applications in natural settings.

## WildScenes Dataset

### Description

The WildScenes dataset is a large-scale benchmark designed for semantic segmentation in natural environments. It includes a wide variety of 2D images that capture complex scenes with different weather conditions, lighting, and terrains.

**Key Features:**

- **Diverse Environments:** Images feature forests and more.
- **Challenging Conditions:** Includes variations in lighting, occlusion, and scale.
- **High Resolution:** Provides detailed images for precise segmentation.

### Dataset Structure

The dataset is organized into directories containing 2D images and their corresponding semantic labels.

```
WildScenes/
│
├── 2D/
│   ├── images/
│   └── labels/
```

## Models Implemented

### UNet

UNet is a convolutional neural network architecture widely used for image segmentation. It features an encoder-decoder structure with skip connections that allow the model to capture both high-level semantic information and fine details, making it particularly effective for segmenting natural scenes.

**UNet Features:**

- **Encoder-Decoder Structure:** Captures detailed context and spatial information.
- **Skip Connections:** Enhance accuracy by preserving spatial details.
- **Versatile Architecture:** Suitable for various image sizes and complexities.

### UNet-ASPP

UNet-ASPP is an extension of the UNet model that incorporates Atrous Spatial Pyramid Pooling (ASPP). This addition enables the model to capture multi-scale features, making it better suited for handling complex patterns in natural environments.

**UNet-ASPP Features:**

- **Multi-Scale Contextual Understanding:** ASPP captures diverse features at different scales.
- **Improved Segmentation:** More effective in handling complex natural environments.
- **Robustness:** Performs well across different scenarios and lighting conditions.

## Usage

Prepare the WildScenes dataset by downloading it from the [WildScenes Website](https://wildscenes.com/dataset) and place it in the designated directory.

### Training

The model is trained for:
	Epochs: 5
  Batch Size: 32
  Learning Rate: 0.0001
  Optimizer: Adam

After 5 epochs, calculated iou is 0.4581 and 0.5195, and the Visualisation of the results, will be in the test phase will be outputted images.

## how to use
Run main.py
Due to limited upload space, the required ckpt.pth file package was uploaded to google.drive, here's the link https://drive.google.com/drive/folders/1wChzRIfiz7cyEBmA4lKJGIR_M5VaWiVb?usp=share_link.
In addition, the file of image and indexLabel need to be save in 





## Results

Use the trained model to do segmentation on test images, the result is statisfactory.
![alt text](image-1.png)

![alt text](image-2.png)



## Conclusion

Using UNet and UNet-ASPP models on the 2D WildScenes dataset highlights the potential of these architectures in advancing image segmentation tasks. These models demonstrate the ability to handle complex scenes with high precision, contributing to advancements in applications such as environmental monitoring and autonomous navigation.

## Future Work

- **Model Optimization:** Enhance models for real-time applications in dynamic environments.
- **Cross-Dataset Evaluation:** Test models on other datasets to assess generalizability.
- **Advanced Techniques:** Explore additional architectures and techniques for further improvements.

## Acknowledgments

We would like to thank the creators of the WildScenes dataset and the open-source community for their contributions and support.
