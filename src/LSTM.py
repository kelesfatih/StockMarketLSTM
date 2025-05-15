import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from src.binance_api_endpoints import MarketDataEndpoints
from src.utils import split_data, sliding_window_with_offset

import neptune

load_dotenv("neptune_credentials.env")
project_name = os.environ.get("NEPTUNE_PROJECT_NAME")
api_key = os.environ.get("NEPTUNE_API_TOKEN")

run = neptune.init_run(project=project_name, api_token=api_key)
params = {
    "input_sz": 1,
    "hidden_sz": 64,
    "n_layers": 3,
    "output_dim": 1,
    "dropout": 0.0,
    "h0": None,
    "c0": None,
    "lr": 0.001,
    "bs": 16,
    "n_epochs": 20
}
run["parameters"] = params


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Fetch Data
asset = "BTCUSDT"
time_frame = "1d"
mde = MarketDataEndpoints()
bitcoin_price_data = mde.fetch_klines(symbol=asset, interval=time_frame, start_str="1 year ago UTC")

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
train_loader = DataLoader(TensorDataset(train_slider[0], train_slider[1]), batch_size=params["bs"], shuffle=False)
test_loader = DataLoader(TensorDataset(X_test), batch_size=params["bs"], shuffle=False)


# Model
class LSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz: int, n_layers: int, output_dim, dropout=0.0):
        super(LSTM, self).__init__()
        self.hidden_sz = hidden_sz
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=input_sz, hidden_size=hidden_sz, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_sz, out_features=output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_sz).to(device)
            c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_sz).to(device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(output[:, -1, :])
        return out, hn, cn


# Model Summary
model = LSTM(params["input_sz"], params["hidden_sz"], params["n_layers"], params["output_dim"], params["dropout"]).to(device)
print(model)
print(summary(model))

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])


def train():
    best_loss = float('inf')
    for epoch in range(params["n_epochs"]):
        model.train()
        running_loss = 0
        with tqdm(train_loader, colour="white", unit="batch") as t:
            for x_batch, y_batch in t:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred, h0, c0 = model(x_batch, h0=params["h0"], c0=params["c0"])

                loss = loss_fn(y_pred, y_batch)
                run["train/batch/loss"].append(loss)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                h0 = h0.detach()
                c0 = c0.detach()

                running_loss += loss.item()
                t.set_description(f"Epoch [{epoch}/{params['n_epochs']}]")
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
run.stop()