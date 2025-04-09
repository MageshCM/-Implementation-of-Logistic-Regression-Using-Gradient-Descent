# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries such as NumPy, Pandas, and Matplotlib.

2.Load the dataset and preprocess it (handle missing values, normalize features, etc.).

3.Define the sigmoid function to map predicted values to a probability between 0 and 1.

4.Initialize parameters (weights and bias) to zeros.

5.Implement the cost function (log loss) to evaluate model performance.

6.Apply Gradient Descent to update weights and bias iteratively to minimize the cost function.

7.Train the model using the training dataset and compute the loss after each iteration.

8.Predict outcomes using the learned weights and compute accuracy.

9.Visualize the loss curve and decision boundary (if applicable).
 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Magesh C M
RegisterNumber:  212223220053
*/

# Program to implement Logistic Regression using Gradient Descent

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost Function
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    return cost

# Gradient Descent
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)
    
    return weights, cost_history

# Data Preparation
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=1)
X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
y = y.reshape(-1, 1)
weights = np.zeros((X.shape[1], 1))

# Training
learning_rate = 0.1
iterations = 1000
weights, cost_history = gradient_descent(X, y, weights, learning_rate, iterations)

# Plotting Cost Function
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Time")
plt.grid(True)
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/45f71c5e-6be0-46af-8b51-7aa5be15dd6b)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

