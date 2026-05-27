import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.timeTools import epoch_diff
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
X, y = load_diabetes(return_X_y= True)
X =  X[:,2].reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2 ,random_state=42)
w = 0
b = 0
Ir = 0.01
epochs = 1000
print(y_train)
for i in range(epochs):
    y_pred = w * X_train + b
    loss = np.mean((y_pred - y_train) ** 2)