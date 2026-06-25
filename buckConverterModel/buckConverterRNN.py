"""
Buck converter RNN: training and evaluation.

Loads the dataset from buckConverterSim.py and trains a
many-to-one RNN to predict output voltage (vO) one switching
period ahead, given a 50-period window of [D, iL, vO].

Architecture (CLAUDE.md spec):
  nn.RNN(input_size=3, hidden_size=32, batch_first=True)
  -> nn.Linear(32, 1)
  out[:, -1, :] only -- many-to-one, not many-to-many.

Once trained, this model can replace the physics simulator
for fast real-time vO prediction inside a control loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import (
    TensorDataset, DataLoader, random_split
)
from pathlib import Path

# Path to the .pt dataset written by buckConverterSim.py.
DATA_PATH = "buckConverterModel/buckConverterData.pt"

# Fraction of total samples withheld for test evaluation;
# not seen by the optimizer during training.
TEST_FRACTION = 0.3

# Number of samples processed together in one forward pass.
# Larger batches are faster but give noisier gradients.
BATCH_SIZE = 32

# Full passes over the entire training dataset.
N_EPOCHS = 20

# Step size for Adam; 1e-3 is Adam's standard default.
LEARNING_RATE = 1e-3

# Dimension of the RNN hidden-state vector.
HIDDEN_SIZE = 64


# Many-to-one RNN that maps a window of converter measurements to the predicted vO one period later. The RNN processes all 50 time steps left-to-right, but only the output at the last time step is passed to the linear layer -- one prediction per sequence, not one prediction per time step.
class BuckRNN(nn.Module):

    # Builds the RNN layer and read-out linear layer, registering both as sub-modules so their parameters appear in model.parameters() for the optimizer. No inputs beyond self. No return value. Sets self.rnn (nn.RNN) and self.fc (nn.Linear).
    def __init__(self):
        super().__init__()
        # Hidden Layer
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        # Output Layer: projects the last time step's hidden state to a single vO value.
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    # Runs the sequence x through self.lstm, selects the
    # last time step's output, and projects to a vO value.
    # x - float32 tensor (batch, seq_len, 3); each position
    #     along dim-1 is one [D, iL, vO] observation
    # Returns float32 tensor (batch, 1): predicted vO [V]
    # one switching period past the end of each window.
    # Does not modify self.lstm or self.fc weights; that is
    # handled by the optimizer in train_epoch.
    def forward(self, x):
        # out shape: (batch, seq_len, HIDDEN_SIZE)
        # h_n shape: (1, batch, HIDDEN_SIZE) -- the final
        # hidden state, discarded here because out[:, -1, :]
        # is identical to h_n[0] for a single-layer RNN.
        lstm_out, _ = self.lstm(x)

        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # This is the many-to-one pattern from CLAUDE.md.
        return self.fc(last_output)


# Loads the .pt file written by make_dataset() in buckConverterSim.py, splits it into train and test sets, and wraps both in DataLoaders that yield mini-batches.
# path          - string or Path to the saved .pt dataset
# test_fraction - fraction of samples reserved for testing;
#                 must be in (0, 1)
# batch_size    - samples per mini-batch for both loaders
# Returns (train_loader, test_loader): DataLoader objects that yield (X_batch, y_batch) pairs each iteration. train_loader shuffles each epoch so the optimizer sees varied mini-batches; test_loader preserves order. Reads from the file at path. Does not modify any global variables or write to any stream.
def load_data(path, test_fraction, batch_size):
    saved   = torch.load(path, weights_only=True)
    X, y    = saved["X"], saved["y"]
    # TensorDataset pairs each X[i] with its label y[i] so DataLoader can sample them together.
    dataset = TensorDataset(X, y)
    n_test  = int(len(dataset) * test_fraction)
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(
        dataset, [n_train, n_test]
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# Runs one complete pass over loader, computing the loss on each mini-batch and updating model weights via Adam.
# model     - BuckRNN instance; its parameters are modified
#             in-place by optimizer.step() every batch
# loader    - DataLoader over the training set; each
#             iteration yields (X_batch, y_batch)
# criterion - nn.MSELoss instance; called as
#             criterion(prediction, target)
# optimizer - Adam optimizer bound to model.parameters();
#             its internal moment estimates are updated
#             in-place each call to optimizer.step()
# device    - "cpu" or "cuda"; batches are moved here
#             before the forward pass
# Returns mean MSE over all batches in this epoch [V^2]. Reads from loader. Modifies model parameters and optimizer state in-place. Does not write to any stream.
def train_epoch(model, loader, criterion, optimizer, device):
    # Training mode enables dropout / batch-norm if present.
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(X_batch)
        loss   = criterion(y_pred, y_batch)
        # Same optimizer sequence as optimWarmup.py:
        # zero -> backward -> step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Measures mean MSE on loader without updating model weights.
# Used to track generalisation after each training epoch.
# model     - BuckRNN instance evaluated without gradients
# loader    - DataLoader over the held-out test set
# criterion - nn.MSELoss instance
# device    - "cpu" or "cuda"
# Returns mean MSE over all batches in loader [V^2].
# Does not modify model parameters or optimizer state.
# Uses torch.no_grad() (same pattern as tensorAutograd.py)
# to skip gradient tracking and reduce memory use.
def evaluate(model, loader, criterion, device):
    # Eval mode disables dropout / batch-norm if present.
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred  = model(X_batch)
            total_loss += criterion(y_pred, y_batch).item()
    return total_loss / len(loader)


if __name__ == "__main__":
    # Detect and use GPU if available; otherwise CPU.
    # Same device-detection pattern as tensorAutograd.py.
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_loader, test_loader = load_data(
        DATA_PATH, TEST_FRACTION, BATCH_SIZE
    )
    n_train = len(train_loader.dataset)
    n_test  = len(test_loader.dataset)

    print(f"Loaded dataset from {DATA_PATH}")
    print(f"RNN with {HIDDEN_SIZE} Hidden Layers, 50 Time Steps, {N_EPOCHS} Epochs. Models [D, iL, vO] to predict vO one switching period ahead.")
    print(f"Train: {n_train:,} samples  Test: {n_test:,}")

    model     = BuckRNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss = evaluate(
            model, test_loader, criterion, device
        )
        # RMSE converts V^2 back to V for physical intuition.
        test_rmse = test_loss ** 0.5
        print(
            f"Epoch {epoch + 1:3d}/{N_EPOCHS}"
            f"  train MSE: {train_loss:.6f} V^2"
            f"  test MSE: {test_loss:.6f} V^2"
            f"  test RMSE: {test_rmse:.4f} V"
            f"  Samples Trained: {n_train*(epoch+1):,}"
            f"  Samples Tested: {n_test*(epoch+1):,}"
        )

    # Persist weights so the model can be loaded for
    # inference without retraining.
    save_path = Path(
        "buckConverterModel/buckConverterRNN.pt"
    )
    torch.save(model.state_dict(), save_path)
    print(f"Saved model -> {save_path}")
