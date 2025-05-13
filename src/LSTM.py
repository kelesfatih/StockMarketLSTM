import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from src.binance_api_endpoints import MarketDataEndpoints
from src.utils import split_data, sliding_window_with_offset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Hyperparameters
hp = {
    "input_size": 1,
    "hidden_size": 64,
    "num_layers": 3,
    "output_dim": 1,
    "h0": None,
    "c0": None,
    "loss": nn.MSELoss(),
    "learning_rate": 0.001,
    "batch_size": 16,
    "num_epochs": 2
}

# Fetch Data
asset = "BTCUSDT"
time_frame = "4h"
mde = MarketDataEndpoints()
bitcoin_price_data = mde.fetch_klines(symbol=asset, interval=time_frame, start_str="5 year ago UTC")

# Split Data
train_raw, test_raw = split_data(bitcoin_price_data, 0.8)

# Scale Data
scaler = StandardScaler()
joblib.dump(scaler, f"../scalers/scaler_{asset}_{time_frame}.joblib")
train = scaler.fit_transform(train_raw)
test = scaler.transform(test_raw)

# Prepare Train Data
train_slider = sliding_window_with_offset(train, 50, 50)

# Prepare Test Data
raw = bitcoin_price_data["Close"][len(bitcoin_price_data) - len(test) - 50:].values
raw = raw.reshape(-1, 1)
raw = scaler.transform(raw)
window_size = 50
X_test = [raw[i - window_size:i, 0] for i in range(window_size, raw.shape[0])]
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test = torch.from_numpy(X_test).type(torch.float32)

# Prepare DataLoaders
train_loader = DataLoader(TensorDataset(train_slider[0], train_slider[1]), batch_size=hp["batch_size"], shuffle=False)
test_loader = DataLoader(TensorDataset(X_test), batch_size=hp["batch_size"], shuffle=False)


# Model
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=hp["input_size"], hidden_size=hp["hidden_size"], num_layers=hp["num_layers"],
                            batch_first=True)
        self.linear = nn.Linear(in_features=hp["hidden_size"], out_features=hp["output_dim"])

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(hp["num_layers"], x.size(0), hp["hidden_size"]).to(device)
            c0 = torch.zeros(hp["num_layers"], x.size(0), hp["hidden_size"]).to(device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(output[:, -1, :])
        return out, hn, cn


# Model Summary
model = LSTM().to(device)
print(summary(model))

# Loss function and optimizer
loss_fn = hp["loss"]
optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])


def train():
    best_loss = float('inf')
    for epoch in range(1, hp["num_epochs"] + 1):
        model.train()
        running_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") as t:
            for x_batch, y_batch in t:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred, h0, c0 = model(x_batch, h0=hp["h0"], c0=hp["c0"])
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                h0 = h0.detach()
                c0 = c0.detach()
                running_loss += loss.item()
                t.set_postfix(loss=loss.item())
        average_loss = running_loss / len(train_loader)
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), "../trained_models/best_model.pth")
            print(f"Saved best model at epoch {epoch} with loss: {average_loss:.4f}")


def test():
    model.load_state_dict(torch.load("../trained_models/best_model.pth"))
    model.eval()
    y_pred_list = []
    with torch.inference_mode():
        for batch in test_loader:
            x = batch[0].to(device)
            outputs, _, _ = model(x)
            y_pred_list.append(outputs)
    y_pred = torch.cat(y_pred_list, dim=0)
    window_size = 50
    y_true_np = raw[window_size:]
    y_true = torch.from_numpy(y_true_np).type(torch.float32).to(device)
    return y_pred, y_true


def plot_results():
    y_pred, y_true = test()
    y_pred_np = y_pred.cpu().detach().numpy().flatten().reshape(-1, 1)
    y_true_np = y_true.cpu().detach().numpy().flatten().reshape(-1, 1)
    y_pred_original = scaler.inverse_transform(y_pred_np).flatten()
    y_true_original = scaler.inverse_transform(y_true_np).flatten()
    df_plot = pd.DataFrame({
        "Index": range(len(y_true_original)),
        "True Value": y_true_original,
        "Predicted Value": y_pred_original
    })
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Index", y="True Value", data=df_plot, label="True")
    sns.lineplot(x="Index", y="Predicted Value", data=df_plot, label="Predicted")
    plt.xlabel("Sample")
    plt.ylabel("Close Price")
    plt.title("LSTM Predictions vs Original True Values")
    plt.legend()
    plt.show()


train()
test()
plot_results()
