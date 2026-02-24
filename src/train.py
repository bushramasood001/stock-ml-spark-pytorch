"""
train.py

# PURPOSE: Train PyTorch model to predict Next_Return and save model + scaler.
"""

import numpy as np
import joblib
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from model import StockReturnNN


# -------------------------------------------------
# PURPOSE: Set seed for reproducibility
# -------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)


# -------------------------------------------------
# PURPOSE: Load your prepared ML dataset (AAPL example)
# NOTE: Here you will paste/load X and y from your notebook export
# -------------------------------------------------
# Example placeholders (replace with your real arrays)
# X = ...  # shape (n, 4)  -> [Daily_Return, Log_Return, MA_7, Volatility_7]
# y = ...  # shape (n,)    -> Next_Return

# ✅ QUICK FIX: If you saved numpy arrays earlier:
# X = np.load("X.npy")
# y = np.load("y.npy")


# -------------------------------------------------
# PURPOSE: Train/Test split (same as notebook)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


# -------------------------------------------------
# PURPOSE: Standardize features (industry practice)
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# -------------------------------------------------
# PURPOSE: Convert to PyTorch tensors
# -------------------------------------------------
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t  = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)


# -------------------------------------------------
# PURPOSE: Initialize model + loss + optimizer
# -------------------------------------------------
model = StockReturnNN(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -------------------------------------------------
# PURPOSE: Training loop
# -------------------------------------------------
epochs = 50
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t)
        train_mse = mean_squared_error(y_train, preds.detach().numpy())
        test_mse  = mean_squared_error(y_test, test_preds.detach().numpy())

        print(f"Epoch {epoch} | Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f}")


# -------------------------------------------------
# PURPOSE: Save trained model (.pt) and scaler (.pkl)
# -------------------------------------------------
torch.save(model.state_dict(), "stock_return_nn.pt")
joblib.dump(scaler, "scaler.pkl")

print("✅ Saved: stock_return_nn.pt and scaler.pkl")
