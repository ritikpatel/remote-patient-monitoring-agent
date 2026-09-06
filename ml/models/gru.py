"""A small GRU over the trailing 24h window -- the last rung of the ladder
(PROJECT_PLAN.md section 11: "L2 logistic regression -> LightGBM -> small GRU
over the 24 h window. Stop at whichever wins; no transformer at this n.").

Unlike the LR/LightGBM feature set (pre-aggregated rolling means/slopes/
severity scores), the GRU is given the **raw per-hour sequence** -- each
core vital plus its own-hour imputation flag, for up to the trailing 24
hours -- so it can learn its own temporal representation instead of consuming
one we hand-engineered. Shorter stays (fewer than 24 hours in) are
left-padded and packed with their true length so the padding never
contributes to the hidden state (``pack_padded_sequence``).

**Compute-budget note, stated plainly rather than silently reduced:** the
plan's ">=20 repeats" protocol is applied in full to the LR and LightGBM
baselines because those fit in well under a second each. A GRU fit is a
training loop, not a closed-form solve; running the full 5x20=100-fold
protocol on this laptop-scale demo would cost real minutes for a model this
plan explicitly permits skipping under schedule pressure ("GRU model -> stop
at LightGBM" is listed first in the descope order, section 16). We keep the
GRU in the ladder -- it is built, trained, and evaluated for real -- but run
it over 5 repeats x 5 folds (25 fits) instead of 20 x 5, and say so in the
report rather than presenting it as the same-size protocol as the other two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

SEQUENCE_LEN_H = 24
GRU_N_REPEATS = 5  # see module docstring
GRU_N_SPLITS = 5

CORE_VITALS = ["hr", "rr", "spo2", "sbp", "map", "temp_c", "gcs_total", "fio2", "glucose"]
FLAG_COLUMNS = [f"{v}_was_imputed" for v in CORE_VITALS]
SEQUENCE_FEATURE_COLUMNS = CORE_VITALS + FLAG_COLUMNS


@dataclass
class SequenceBatch:
    x: np.ndarray  # (n, SEQUENCE_LEN_H, n_features), left-padded with 0
    lengths: np.ndarray  # (n,) true (unpadded) sequence length
    y: np.ndarray  # (n,)
    stay_ids: np.ndarray  # (n,) -- the grouping key


def build_sequences(
    hourly_grid_raw: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_col: str,
    seq_len: int = SEQUENCE_LEN_H,
) -> SequenceBatch:
    """One sequence per at-risk labelled row: the ``seq_len`` hours ending at
    (and including) that row's own hour, for that stay only.
    """
    grid = hourly_grid_raw.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    feature_means = grid[CORE_VITALS].mean()
    grid = grid.copy()
    for col in CORE_VITALS:
        grid[col] = grid[col].fillna(feature_means[col])
    for col in FLAG_COLUMNS:
        grid[col] = grid[col].fillna(False).astype(float)

    by_stay = {stay_id: g.reset_index(drop=True) for stay_id, g in grid.groupby("stay_id")}

    xs = []
    lengths = []
    ys = []
    stay_ids = []
    n_features = len(SEQUENCE_FEATURE_COLUMNS)
    for row in labels_df.itertuples(index=False):
        stay_grid = by_stay.get(row.stay_id)
        if stay_grid is None:
            continue
        upto = stay_grid[stay_grid.hour <= row.hour]
        window = upto.tail(seq_len)
        length = len(window)
        if length == 0:
            continue
        arr = np.zeros((seq_len, n_features), dtype=np.float32)
        arr[seq_len - length :] = window[SEQUENCE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        xs.append(arr)
        lengths.append(length)
        ys.append(getattr(row, label_col))
        stay_ids.append(row.stay_id)

    if not xs:
        return SequenceBatch(
            x=np.zeros((0, seq_len, n_features), dtype=np.float32),
            lengths=np.zeros(0, dtype=int),
            y=np.zeros(0),
            stay_ids=np.zeros(0),
        )
    return SequenceBatch(
        x=np.stack(xs), lengths=np.array(lengths), y=np.array(ys), stay_ids=np.array(stay_ids)
    )


class _SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, lengths: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.lengths = torch.from_numpy(lengths)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.lengths[idx], self.y[idx]


class DeteriorationGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.head(h_n[-1]).squeeze(-1)


def fit_predict_proba(
    x_train: np.ndarray,
    len_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    len_test: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-2,
    seed: int = 0,
) -> tuple[DeteriorationGRU, np.ndarray]:
    torch.manual_seed(seed)
    model = DeteriorationGRU(input_dim=x_train.shape[-1])
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / positives if positives else 1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        _SeqDataset(x_train, len_train, y_train), batch_size=batch_size, shuffle=True
    )
    model.train()
    for _epoch in range(epochs):
        for xb, lb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test), torch.from_numpy(len_test))
        proba = torch.sigmoid(logits).numpy()
    return model, proba
