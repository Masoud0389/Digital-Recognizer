# Digital Recognizer - Kaggle Dataset Analysis

This repository contains Python code for analyzing and recognizing handwritten digits using the MNIST dataset from Kaggle. The code utilizes Convolutional Neural Networks (CNN) and compares the performance with traditional Machine Learning (ML) algorithms such as K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Decision Tree.

## Requirements
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow

## Usage
1. Download the MNIST dataset from Kaggle and save it in the appropriate directory.
2. Update the `train_data` and `test_data` variables in the code with the correct paths to the dataset files.
3. Run the code to perform the following steps:
   - Load and preprocess the MNIST dataset.
   - Create a CNN model and train it using the training data.
   - Evaluate the CNN model on the test data and calculate its accuracy.
   - Predict the test set labels using the CNN model.
   - Compare the performance of traditional ML algorithms (KNN, SVM, Decision Tree) with the CNN model.
   - Calculate the accuracy scores and plot confusion matrices for each algorithm.
   - Predict the labels for the submission dataset and save the results in a CSV file.
4. Observe the accuracy scores, confusion matrices, and the generated submission file to assess the performance of the CNN model and the traditional ML algorithms.

Feel free to modify the code according to your specific needs, such as changing the dataset, adjusting the CNN architecture, or experimenting with different ML algorithms.

## Acknowledgments
The code in this repository was created based on the MNIST dataset available on Kaggle. Special thanks to the original dataset contributors.


Remember to replace the placeholder information, such as dataset paths and file names, with the correct details for your specific case.
