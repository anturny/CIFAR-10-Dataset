# CIFAR-10-Dataset
- This project demonstrates the fundamentals of machine learning through the classification of images using the CIFAR-10 dataset. By training a deep learning model, a convolutional neural network in this case, we aim to teach the system to recoginize and categorize small color images into ten different classes, such as airplanes, cars, and animals. The goal is to develop a model capable of accurately predicting the category of new and unseen images based on learned features.

# Table of Contents
- [Implementation](#implementation)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
- [References](#references)

## Implementation
- This CIFAR-10 dataset focuses on image classification by using deep learning. CIFAR-10 is a widely used dataset consisting of 60,000 small color images categorized into 10 different classes, such as airplanes, cars, birds, and animals. The purpose of this dataset is to train and evaluate convolutional neural networks to accurately recognize and classify images into their respective categories. By utilizing this dataset, the model learns to extract meaningful features from images, enabling it to make predictions on unseen data with high accuracy. 

## Requirements
- Visual Studio Code (Software)
- Python Language on Computer (3.12.0)
- GitBash (Optional)

- This project is designed to run in a VSCode terminal using a Python environment.

Use 'pip install -r requirements.txt' tp install the following dependencies:
```
tensorflow==2.20.0
numpy==2.3.3
matplotlib==3.10.6
keras==3.11.3
```

## How to Use
- To run this code, you will need to have a Python environment installed on your computer. It is recommended to use Visual Studio Code as this Python script was written and ran in VSCode. GitBash is also recommended in order to synchronize your VSCode with GitHub.
- In GitHub, click on the green icon labeled "<> CODE" on the top of this page and copy the HTTPS link.
- In VSCode, click on "Clone Git Repository" and paste the copied link from GitHub.
- In the search bar, type in "Python: Create Environment" and then select a preferred environment. This code used .venv as the virtual environment.
- When the virtual environment is open (appears as .venv in the list of items in the left menu), you may navigate to the [CIFAR Dataset.py](/src/CIFAR%20Dataset.py) file and select it. At this point, you may open your terminal and install the pip requirements for the necessary libraries in order to execute the code. Then, you may hit "Run" on the top right hand corner to execute the code.

- The CIFAR Dataset can be found in the reference link below, but tensorflow, a necessary library, already includes the CIFAR dataset which can be found in the code via:
```
cifar10 = tf.keras.datasets.cifar10
```

## References
- [1]GeeksforGeeks, “CIFAR10 Image Classification in TensorFlow,” GeeksforGeeks, Apr. 29, 2021. https://www.geeksforgeeks.org/deep-learning/cifar-10-image-classification-in-tensorflow/
