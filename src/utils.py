import torch
import numpy as np


def split_data(price_data, training_ratio):
    train_size = int(training_ratio * len(price_data))
    train_raw_f = price_data[:train_size][["Close"]]
    test_raw_f = price_data[train_size:][["Close"]]
    return train_raw_f, test_raw_f


def sliding_window_with_offset(price_data, N, offset):
    X, y = [], []
    for i in range(offset, len(price_data)):
        X.append(price_data[i - N: i])
        y.append(price_data[i])
    array_x = np.array(X)
    array_y = np.array(y)
    tensor_x = torch.from_numpy(array_x).type(torch.float32)
    tensor_y = torch.from_numpy(array_y).type(torch.float32)
    return tensor_x, tensor_y
