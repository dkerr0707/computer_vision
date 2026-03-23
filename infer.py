import torch
from train import SimpleNet, INPUT_SIZE

MODEL_PATH = "model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_PATH}")
print(f"Using device: {device}")

# Example: run inference on random input
torch.manual_seed(0)
x = torch.randn(5, INPUT_SIZE).to(device)

with torch.no_grad():
    preds = model(x)

for i, pred in enumerate(preds):
    print(f"Sample {i+1}: {pred.item():.4f}")
