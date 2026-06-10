"""Lightweight PyTorch MLP regressor for prediction-table experiments."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cleaned.astype("float32")


def predict_with_mlp(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    epochs: int = 8,
    batch_size: int = 2048,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_train = _clean_features(train_x)
    x_test = _clean_features(test_x)
    y_train = pd.to_numeric(train_y, errors="coerce").astype("float32")
    mask = y_train.notna().to_numpy()
    x_train = x_train.loc[mask]
    y_train = y_train.loc[mask]

    scaler = StandardScaler()
    train_values = scaler.fit_transform(x_train).astype("float32")
    test_values = scaler.transform(x_test).astype("float32")

    dataset = TensorDataset(torch.from_numpy(train_values), torch.from_numpy(y_train.to_numpy()))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TabularMLP(train_values.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_values), batch_size):
            batch = torch.from_numpy(test_values[start : start + batch_size]).to(device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds).astype("float64")


def build_prediction_frame(test_data: pd.DataFrame, test_y: pd.Series, pred_y: np.ndarray) -> pd.DataFrame:
    output_columns = [
        "trade_date",
        "name",
        "stock_code",
        "pred_prob",
        "std_his_high",
        "10d_yield_rate",
        "st_type",
        "post_high",
        "post_close",
        "post2_close",
        "post2_high",
        "open3_yield_rate",
        "open2_yield_rate",
        "limit_times",
        "close",
        "pre_close",
        "post_open",
        "post2_open",
        "post3_open",
        "post4_open",
        "post5_open",
        "post6_open",
        "post12_open",
        "industry_encode",
        "atr_qfq",
        "close_rate",
    ]
    label = getattr(test_y, "name", None)
    if label not in output_columns and label in test_data.columns:
        output_columns.append(label)
    result = test_data.copy()
    result["pred_prob"] = pred_y
    return result[[col for col in output_columns if col in result.columns]]
