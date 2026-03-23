import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_mnist import MNISTNet

MODEL_PATH = "mnist_model.pth"
NUM_SAMPLES = 10000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MNISTNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
loader = DataLoader(test_set, batch_size=NUM_SAMPLES, shuffle=True)

X_batch, y_batch = next(iter(loader))
X_batch = X_batch.to(device)

with torch.no_grad():
    preds = model(X_batch).argmax(1).cpu()

print(f"\n{'Sample':<8} {'Ground Truth':<14} {'Predicted':<10} {'Correct'}")
print("-" * 44)
for i, (gt, pred) in enumerate(zip(y_batch, preds)):
    correct = "yes" if gt == pred else "no"
    print(f"{i+1:<8} {gt.item():<14} {pred.item():<10} {correct}")

accuracy = (preds == y_batch).float().mean().item()
print(f"\nAccuracy on {NUM_SAMPLES} samples: {accuracy * 100:.1f}%")
