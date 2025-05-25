# Semantic Segmentation Project

## Description

This project performs semantic segmentation on a dataset of forest scenes using the DeepLabV3 model. The goal is to classify each pixel in an image into one of several categories, providing a detailed understanding of the scene. The project uses two different encoder models: ResNet34 and MobileNetV2.

project/
├── dataset.py
├── mobilenetv2.py
├── resnet34.py
├── train.py
├── evaluate.py
├── utils.py
├── main.py
├── ws.py
├── requirements.txt
├── README.md
├── notebooks/
│   ├── resnet34.ipynb
│   └── mobilenetv2.ipynb


## Files

- `dataset.py`: Defines the dataset class, `WSdataset`, which loads images and corresponding masks, preprocesses them, and returns them in a format suitable for training and evaluation.
- `mobilenetv2.py`: Defines the DeepLabV3 model with MobileNetV2 encoder.
- `resnet34.py`: Defines the DeepLabV3 model with ResNet34 encoder.
- `train.py`: Contains the training loop function, `train()`, which handles the training of the model on the dataset.
- `evaluate.py`: Contains the evaluation functions:
  - `test()`: Tests the model on the validation dataset and calculates the loss.
  - `get_metrics2()`: Evaluates the model's performance using metrics such as classification report and Mean Intersection over Union (mIoU).
- `utils.py`: Utility functions for loading and preparing data, including:
  - `load_image_paths()`: Loads image paths from the dataset directory.
  - `prepare_data_dict()`: Prepares a dictionary of image and mask paths for DataFrame creation.
- `main.py`: Main script to run the training and evaluation. It uses command-line arguments to select the model type (ResNet34 or MobileNetV2) and handles the overall workflow.
- `ws.py`: Script to quickly demonstrate the execution process.
- `requirements.txt`: Lists all the required Python packages for the project.
- `README.md`: Provides an overview of the project, setup instructions, and usage details.
- `notebooks/`: Directory containing Jupyter notebooks for visualizing results:
  - `resnet34.ipynb`: Notebook for running and visualizing results using the ResNet34 model.
  - `mobilenetv2.ipynb`: Notebook for running and visualizing results using the MobileNetV2 model.
- `data/`: Directory containing the dataset:
  - `K-03/`: Subdirectory containing images and corresponding masks.

## Setup

### Requirements

Install the required packages using `pip`:

```sh
pip install -r requirements.txt



### To run with ResNet34:
```sh
python main.py --model resnet34 > resnet34.log 2>&1
### To run with Mobilenetv2:
```sh
python main.py --model mobilenetv2 > mobilenetv2.log 2>&1

### To quickly demonstrate the execution process, use the following command (as the video demo shows):
```sh
python ws.py > demo.log 2>&1  ### log file name "demo" could change to resnet34 or mobilenetv2.

### To view the results, you can open the respectively log file:
cat resnet34.log
cat mobilenetv2.log


# train,test process cite from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Deeplabv3 model cite from https://github.com/qubvel-org/segmentation_models.pytorch.git