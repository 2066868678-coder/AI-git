import numpy as np
y_true = np.array([12.3, 15.1, 9.8, 20.5, 14.2])
y_pred = np.array([11.8, 15.9, 10.2, 19.7, 14.6])
mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)
print(f'RMSE= {rmse:.4f}')
