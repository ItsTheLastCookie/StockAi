import os
import pandas as pd
import numpy as np


path = r"/kaggle/input/nvidia-corporation-nvda-stock-price/" 
file_name = "NVDA(2).csv"
file_path = os.path.join(path, file_name)

data = pd.read_csv(file_path)

print(data)
learning_rate = 0.1
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def compute_gradients(X, y, theta):
    n = X.shape[0]
    gradients = (1/n) * X.T.dot(X.dot(theta) - y)
    return gradients
def gradient_descent(X, y, theta, learning_rate, iterations=1000):
    cost_history = []
    for i in range(iterations):

        y_pred = X.dot(theta)
        
        cost = mse(y, y_pred)

        cost_history.append(cost)

        gradients = compute_gradients(X, y, theta)

        theta = theta - learning_rate * gradients
        
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost}")
    
    return theta, cost_history
theta = np.zeros(2)
X = np.c_[np.ones(data.shape[0]), data["Close"]]
y_pred = X.dot(theta)
epochs = 10
for i in range(epochs):
    y = data["Close"]
    gradients = compute_gradients(X, y, theta)
    theta = theta - learning_rate * gradients
    y_pred = X.dot(theta)
    print(mse(y, y_pred))
result = y - y_pred
print(result)
buy = 0 
sell = 0
for res in result:
    if res > 0:
        buy += 1
    else:
        sell += 1
print(f"Buy: {buy}, Sell: {sell}")
