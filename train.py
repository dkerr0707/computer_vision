import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Hyperparameters ---
INPUT_SIZE = 20
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 20


# --- Model ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x):
        return self.net(x)


# --- Training loop ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item() * len(X_batch)
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    # --- Synthetic dataset (regression task) ---
    torch.manual_seed(42)
    X = torch.randn(1000, INPUT_SIZE)
    y = X[:, :OUTPUT_SIZE] * 2.5 + 1.0  # simple linear target with noise
    y += torch.randn_like(y) * 0.1

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:>3}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    print("\nTraining complete.")
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")
