"""
model/lstm_model.py
────────────────────
LSTM + Monte Carlo Dropout forecasting model.

Architecture
────────────
  Input  : (batch, window, 1)   — normalised CPU utilisation sequence
  Layers : stacked LSTM -> dropout -> fully-connected head
  Output : (batch, horizon)     — multi-step point forecast

MC Dropout
──────────
PyTorch disables dropout in eval mode by default.  To keep dropout active
during inference (MC Dropout), call `enable_mc_dropout(model)` before running
stochastic forward passes.  The context manager `mc_dropout_ctx(model)` handles
this automatically, restoring the original mode on exit.
"""

from __future__ import annotations

import contextlib
import torch
import torch.nn as nn


class LSTMForecast(nn.Module):
    """Multi-step LSTM forecaster with MC Dropout support.

    Parameters
    ----------
    input_size  : feature dimension (1 for univariate CPU util).
    hidden_size : LSTM hidden state dimension.
    num_layers  : number of stacked LSTM layers.
    dropout     : dropout probability applied between LSTM layers and before
                  the output head.  Kept active at inference via MC Dropout.
    horizon     : number of future timesteps to predict.
    """

    def __init__(
        self,
        input_size:  int = 1,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.3,
        horizon:     int = 15,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.horizon     = horizon

        # batch_first=True -> input/output shape: (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Explicit dropout layer so we can keep it active at inference
        self.dropout = nn.Dropout(p=dropout)

        # Output head: maps last hidden state -> horizon predictions
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, window, input_size)

        Returns
        -------
        torch.Tensor, shape (batch, horizon)
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Use only the last timestep's hidden state
        last_hidden = lstm_out[:, -1, :]           # (batch, hidden_size)
        dropped = self.dropout(last_hidden)
        return self.fc(dropped)                    # (batch, horizon)


# ── MC Dropout helpers ────────────────────────────────────────────────────────

def enable_mc_dropout(model: nn.Module) -> None:
    """Set all Dropout layers to training mode (keeps dropout active).

    Call this after `model.eval()` to enable stochastic forward passes.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@contextlib.contextmanager
def mc_dropout_ctx(model: nn.Module):
    """Context manager: run stochastic MC-Dropout forward passes.

    Puts the model in eval mode (disables BatchNorm updates, etc.) but keeps
    Dropout layers active.  Restores the original training flag on exit.

    Usage
    -----
    >>> with mc_dropout_ctx(model):
    ...     for _ in range(n_samples):
    ...         preds.append(model(x))
    """
    was_training = model.training
    model.eval()
    enable_mc_dropout(model)
    try:
        yield model
    finally:
        model.train(was_training)


# ── factory ───────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> LSTMForecast:
    """Construct an LSTMForecast from the config dict."""
    m = cfg["model"]
    return LSTMForecast(
        input_size=1,
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        horizon=m["horizon"],
    )


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    print(model)

    window  = cfg["preprocessing"]["window_size"]
    horizon = cfg["model"]["horizon"]
    batch   = torch.randn(4, window, 1)

    # Normal forward pass
    model.eval()
    with torch.no_grad():
        out = model(batch)
    print(f"\nEval forward pass  -> output shape: {out.shape}")

    # MC Dropout forward pass
    with mc_dropout_ctx(model):
        with torch.no_grad():
            out_mc = model(batch)
    print(f"MC-Dropout forward -> output shape: {out_mc.shape}")
    print("\n[OK] LSTMForecast model OK")
