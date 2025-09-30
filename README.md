# CIFAR-10-Dataset
- This project demonstrates the basics of machine 
learning using the CIFAR-10 Dataset

# Table of Contents
- [Implementation](#implementation)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
- [Error Handling](#error-handling)
- [References](#references)

## Implementation
- This model consists of over 60,000 models of 10 different categories such as Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck. These images are 32x32 with 50,000 total train images and 10,000 test images. This model will be used to 50 epochs of which we reduce the pixel values and then utilize a convolution neural network to train our model  with adam optimizer and sparse_categorical_crossentropy as the loss function.

## Requirements
- This project is designed to run in a VSCode 
terminal using a Python environment.

## How to Use
- To run this code, you will need to have a Python 
environment installed on your computer. You can download "cifar_10_dataset.py" into a folder, and open the folder within VSCode.
- The CIFAR-10 dataset is automatically downloaded and loaded in the script, so no external dataset is required.

## Error Handling
- This project does not have any error handling.

## References
- [1]GeeksforGeeks, “CIFAR10 Image Classification in TensorFlow,” GeeksforGeeks, Apr. 29, 2021. https://www.geeksforgeeks.org/deep-learning/cifar-10-image-classification-in-tensorflow/