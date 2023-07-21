# Bharat-task-3
# Handwritten Digit Recognition using CNN (Task-3 of Bharat Internship)

## Problem Statement

A handwritten digit recognition system aims to detect scanned images of handwritten digits (0 to 9) using machine learning techniques. In this project, we tackle this task by implementing a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset.

## Overview

This repository contains the code for a Handwritten Digit Recognition system implemented using Convolutional Neural Networks (CNN). It is Task-3 of the Bharat Internship, demonstrating the use of CNNs for image recognition tasks.

## Project Description

In this project, we build and train a CNN model using the MNIST dataset, a popular dataset of handwritten digits. The model is trained to recognize digits from 0 to 9 based on the input images. We use TensorFlow and Keras libraries to create and train the CNN.

The MNIST dataset consists of 28x28 grayscale images of handwritten digits along with their corresponding labels. The CNN model learns to extract meaningful features from the images and classifies them into their respective digits.

## Project Structure

- `mnist_digit_recognition.ipynb`: Jupyter Notebook containing the code for data preprocessing, building the CNN model, training, and evaluation.
- `training_data/`: Folder containing the preprocessed training dataset (train_images.npy and train_labels.npy).
- `test_data/`: Folder containing the preprocessed test dataset (test_images.npy and test_labels.npy).

## Instructions

To run the code and reproduce the results:

1. Install the required libraries mentioned in `requirements.txt`.
2. Run the Jupyter Notebook `mnist_digit_recognition.ipynb` to train the CNN model and evaluate its performance.
3. To make predictions on new images, replace `'path_to_new_image.jpg'` in the prediction code with the path to the new image you want to test.

## License

This project is licensed under the [MIT License](LICENSE).

## Credits

The MNIST dataset is sourced from [TensorFlow Datasets](https://www.tensorflow.org/datasets).

## Acknowledgments

Special thanks to the Bharat Internship program for providing this opportunity to learn and work on exciting projects!

For any questions or inquiries, feel free to contact [jakkulakeerthirani2@gmail.com](jakkulakeerthirani2@gmail.com).
