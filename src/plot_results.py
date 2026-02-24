# PURPOSE: Plot Actual vs Predicted and compute simple error metrics

# PURPOSE: Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# PURPOSE: Load actual and predicted arrays
y_true = np.load("outputs/y.npy")        # shape (6467,1)
y_pred = np.load("outputs/y_pred.npy")   # shape (6467,1)

# PURPOSE: Flatten to 1D for metrics/plotting
y_true_1d = y_true.reshape(-1)
y_pred_1d = y_pred.reshape(-1)

# PURPOSE: Compute metrics
mse = mean_squared_error(y_true_1d, y_pred_1d)
mae = mean_absolute_error(y_true_1d, y_pred_1d)

print("MSE:", mse)
print("MAE:", mae)

# PURPOSE: Plot last N points (easy visual check)
N = 200
plt.figure()
plt.plot(y_true_1d[-N:], label="Actual")
plt.plot(y_pred_1d[-N:], label="Predicted")
plt.title(f"Actual vs Predicted (last {N} points)")
plt.xlabel("Time index (last window)")
plt.ylabel("Next_Return")
plt.legend()
plt.show()
