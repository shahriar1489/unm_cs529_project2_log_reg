# Logistic Regression for Music Genre Classification
This repository contains my implementation of logistic regression with gradient descent, applied to a music genre classification problem using a dataset of pre-processed audio features. After training and testing the model, it achieves an accuracy of 71% or above on both the training and test sets.

## Features of the Implementation:
* Gradient Descent Optimization: The logistic regression model is trained using gradient descent with optional L2 regularization to prevent overfitting.
* Softmax Activation: The model outputs class probabilities using the softmax function, and the predicted class is the one with the highest probability.
* Cross-Entropy Loss: The loss function is based on cross-entropy, which is optimized during training.
* Performance: After processing the dataset, the model consistently achieves an accuracy of 71% or higher on both the training and test sets.
## Dataset:
The model has been tested on a pre-processed version of a music genre dataset, which contains features extracted from audio files representing 10 different genres. These features include Mel-frequency cepstral coefficients (MFCCs), chroma, tempo, and other audio characteristics that are useful for genre classification.

## Results:
Training Accuracy: 71%+
Test Accuracy: 71%+
