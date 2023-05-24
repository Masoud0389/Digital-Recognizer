# Digital Recognizer: Comparing CNN with KNN, SVM, and Decision Tree Models

This repository contains the code for a digital recognizer that compares the performance of Convolutional Neural Networks (CNN) with traditional machine learning models such as K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and decision trees.

## Description
The digital recognizer is designed to classify images of handwritten digits into their respective numerical values (0-9). The CNN architecture is used for its effectiveness in image recognition tasks. It utilizes convolutional layers to extract relevant features, followed by pooling layers for downsampling and fully connected layers for classification.

The project includes training and testing the CNN model on a labeled dataset of handwritten digits. The dataset is preprocessed and split into training and testing sets. The CNN model is trained using the training set and its parameters are optimized through backpropagation and gradient descent.

After training, the CNN model is evaluated on an independent testing set, and metrics such as accuracy, precision, recall, and F1-score are computed. These metrics provide insights into the model's performance in classifying handwritten digits.

To compare the performance of the CNN model with traditional machine learning models (KNN, SVM, decision trees), the dataset undergoes similar preprocessing steps. Each model is trained on the training set and evaluated on the testing set using the same metrics.

The project aims to determine which model performs the best in terms of accuracy and overall classification performance. By comparing the results of the CNN model with traditional machine learning models, valuable insights can be gained regarding the suitability and effectiveness of CNN for digital recognition tasks.

## Prerequisites
- Python (3.0 or higher)
- TensorFlow (2.0 or higher)
- Scikit-learn (0.24 or higher)

## Installation
1. Clone this GitHub repository:
```bash
git clone https://github.com/your-username/digital-recognizer.git
cd digital-recognizer
```

## Usage
1. Place your labeled dataset of handwritten digits in the `/kaggle/input/digit-recognizer/train.csv` directory.
2. Run the script to train and evaluate the models:
```bash
python main.py
```
3. The results will be displayed in the console, including the accuracy, precision, recall, and F1-score for each model.

## Contributing
Contributions to this project are welcome. Please follow these guidelines when contributing:
- Fork the repository.
- Create a new branch for your feature/bug fix.
- Commit your changes and push the branch.
- Submit a pull request explaining the changes you've made.

## Kaggle Competition
For an extended evaluation and comparison, you can also find this project on Kaggle. Visit the competition page [here](https://www.kaggle.com/competitions/digit-recognizer) to participate and explore the performance of different models.

## Acknowledgments
We would like to thank the creators of TensorFlow and Scikit-learn for providing the necessary tools and libraries for this project.
