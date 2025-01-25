# Uber-Customer-Reviews-Dataset-2024-

* Sentiment Analysis using Deep Learning

Description:
This program uses the Uber Customer Reviews dataset (12,000+ reviews) from the Google Play Store to perform sentiment analysis. It applies two optimization techniques—Stochastic Gradient Descent (SGD) and Mini-Batch Gradient Descent—to train a neural network for predicting review scores. The program then compares the performance of both techniques using Mean Squared Error (MSE) and visualizes the training loss curves.

By: Jharana Adhikari and Yasaswin Palukuri
Steps:
Step 1: Install necessary libraries (tensorflow, scikit-learn, matplotlib) for data handling, model building, and visualization.
Step 2: Import required libraries for data preprocessing, model building, and evaluation (such as pandas, numpy, tensorflow).
Step 3: Load the Uber customer reviews dataset, clean the data, and preprocess it. This includes: - Encoding categorical text data (e.g., reviews) using techniques like LabelEncoder or TF-IDF. - Handling missing values and preparing features (e.g., review content, user ratings).
Step 4: Split the data into features (e.g., text data or numeric values) and target (e.g., ratings or sentiment labels).
Step 5: Split the data into training and testing sets using train_test_split (80% for training and 20% for testing).
Step 6: Build a neural network model using TensorFlow/Keras, with layers like:
Input Layer (depending on the number of features),
Hidden Layers (e.g., using ReLU activation),
Output Layer (for regression tasks, use a linear activation function).
Step 7: Compile and train the model using Stochastic Gradient Descent (SGD) with a batch size of 1.
Step 8: Compile and train the model using Mini-Batch Gradient Descent with a batch size of 32.
Step 9: Evaluate both models' performances using Mean Squared Error (MSE) to assess how well each optimization technique predicts the reviews' ratings.
Step 10: Visualize and compare the training loss curves for both SGD and Mini-Batch Gradient Descent to understand the learning dynamics of both techniques.

