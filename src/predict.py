# PURPOSE: Load saved model + scaler, then make predictions on X.npy

# PURPOSE: Import required libraries
import numpy as np
import joblib
import torch

# PURPOSE: Import the same model class used during training
from model import StockReturnNN

# PURPOSE: Set paths to saved files
X_PATH = "outputs/X.npy"
MODEL_PATH = "stock_return_nn.pt"
SCALER_PATH = "scaler.pkl"

# PURPOSE: Load X features
X = np.load(X_PATH)

# PURPOSE: Load scaler and apply same scaling as training
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

# PURPOSE: Convert to torch tensor
X_t = torch.tensor(X_scaled, dtype=torch.float32)

# PURPOSE: Rebuild model with same input size (4 features) and load weights
model = StockReturnNN(input_dim=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# PURPOSE: Predict (no gradients needed)
with torch.no_grad():
    y_pred = model(X_t).numpy()

# PURPOSE: Save predictions for later use
np.save("outputs/y_pred.npy", y_pred)

# PURPOSE: Print quick check
print("✅ Saved predictions to outputs/y_pred.npy")
print("y_pred shape:", y_pred.shape)
print("First 5 predictions:", y_pred[:5].reshape(-1))
