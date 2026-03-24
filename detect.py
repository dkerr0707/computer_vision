import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE    = 224        # VOC images resized to 224×224
NUM_CLASSES = 20         # Pascal VOC has 20 object classes
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3
BBOX_LAMBDA = 1.0        # weight on bbox regression loss vs classification loss


# ── Dataset ──────────────────────────────────────────────────────────────────
class VOCDataset(Dataset):
    """
    Wraps torchvision VOCDetection (Pascal VOC 2012).

    For each image the largest non-difficult bounding box is selected as the
    single detection target, keeping compatibility with DetectorCNN's dual-head
    output (one class + one box per image).

    Labels:
        cls   – int in [0, 19]  (VOC class index)
        bbox  – float tensor [x_min, y_min, x_max, y_max], normalised to [0, 1]
    """

    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    ]

    def __init__(self, root='./data', image_set='train', download=True):
        self.voc = torchvision.datasets.VOCDetection(
            root=root, year='2012', image_set=image_set, download=download,
        )
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]
        orig_w, orig_h = img.size   # PIL gives (W, H)

        img_tensor = self.transform(img)

        objects = target['annotation']['object']
        if isinstance(objects, dict):   # single object stored as dict, not list
            objects = [objects]

        best_area = -1.0
        best_cls  = 0
        best_bbox = [0.0, 0.0, 1.0, 1.0]

        for obj in objects:
            name = obj['name']
            if name not in self.CLASSES:
                continue
            if obj.get('difficult', '0') == '1':
                continue

            b = obj['bndbox']
            x_min = float(b['xmin']) / orig_w
            y_min = float(b['ymin']) / orig_h
            x_max = float(b['xmax']) / orig_w
            y_max = float(b['ymax']) / orig_h

            area = (x_max - x_min) * (y_max - y_min)
            if area > best_area:
                best_area = area
                best_cls  = self.CLASSES.index(name)
                best_bbox = [x_min, y_min, x_max, y_max]

        bbox = torch.tensor(best_bbox, dtype=torch.float32)
        return img_tensor, best_cls, bbox


# ── Model ─────────────────────────────────────────────────────────────────────
class DetectorCNN(nn.Module):
    """
    Shared CNN backbone with two task-specific heads:
      • cls_head  – predicts class logits (CrossEntropyLoss)
      • bbox_head – predicts [x_min, y_min, x_max, y_max] (SmoothL1Loss)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # Feature extractor: 4 conv blocks
        # Input:          (B,   3, 224, 224)
        # After block 1:  (B,  32, 112, 112)
        # After block 2:  (B,  64,  56,  56)
        # After block 3:  (B, 128,  28,  28)
        # After block 4:  (B, 256,  14,  14)
        # After avgpool:  (B, 256,   7,   7) → flatten → 12544
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.cls_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid(),   # keep predictions in [0, 1] to match normalised targets
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)   # flatten
        cls_logits = self.cls_head(feat)
        bbox_pred  = self.bbox_head(feat)
        return cls_logits, bbox_pred


# ── IoU utility ───────────────────────────────────────────────────────────────
def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union for two boxes.

    Args:
        box_a, box_b: tensors of shape (..., 4) with format [x_min, y_min, x_max, y_max]

    Returns:
        IoU scalar (or tensor matching the batch prefix shape)
    """
    inter_x0 = torch.max(box_a[..., 0], box_b[..., 0])
    inter_y0 = torch.max(box_a[..., 1], box_b[..., 1])
    inter_x1 = torch.min(box_a[..., 2], box_b[..., 2])
    inter_y1 = torch.min(box_a[..., 3], box_b[..., 3])

    inter_w = (inter_x1 - inter_x0).clamp(min=0)
    inter_h = (inter_y1 - inter_y0).clamp(min=0)
    inter_area = inter_w * inter_h

    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area.clamp(min=1e-6)


# ── NMS utility ───────────────────────────────────────────────────────────────
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression: removes redundant overlapping boxes.

    In a real multi-object detector the network produces many candidate boxes.
    NMS keeps only the highest-scoring box among those that overlap significantly.

    Args:
        boxes:         tensor (N, 4) [x_min, y_min, x_max, y_max]
        scores:        tensor (N,)   confidence score for each box
        iou_threshold: boxes with IoU > this value are suppressed

    Returns:
        kept: list of indices of boxes to keep
    """
    # sort boxes by descending score
    order = scores.argsort(descending=True)
    kept = []

    while order.numel() > 0:
        # always keep the highest-scoring remaining box
        best = order[0].item()
        kept.append(best)

        if order.numel() == 1:
            break

        # compute IoU of the best box against all remaining boxes
        rest = order[1:]
        iou = compute_iou(boxes[best].unsqueeze(0), boxes[rest])  # shape (N-1,)

        # keep only boxes with low overlap
        order = rest[iou <= iou_threshold]

    return kept


# ── Training / evaluation loops ───────────────────────────────────────────────
cls_loss_fn  = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_iou  = 0.0

    for imgs, cls_targets, bbox_targets in loader:
        imgs         = imgs.to(device)
        cls_targets  = cls_targets.to(device)
        bbox_targets = bbox_targets.to(device)

        optimizer.zero_grad()
        cls_logits, bbox_pred = model(imgs)

        # multi-task loss: classification + weighted bbox regression
        loss = cls_loss_fn(cls_logits, cls_targets) + BBOX_LAMBDA * bbox_loss_fn(bbox_pred, bbox_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_iou  += compute_iou(bbox_pred.detach(), bbox_targets).sum().item()

    n = len(loader.dataset)
    return total_loss / n, total_iou / n


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0

    with torch.no_grad():
        for imgs, cls_targets, bbox_targets in loader:
            imgs         = imgs.to(device)
            cls_targets  = cls_targets.to(device)
            bbox_targets = bbox_targets.to(device)

            cls_logits, bbox_pred = model(imgs)
            loss = cls_loss_fn(cls_logits, cls_targets) + BBOX_LAMBDA * bbox_loss_fn(bbox_pred, bbox_targets)

            total_loss += loss.item() * imgs.size(0)
            total_iou  += compute_iou(bbox_pred, bbox_targets).sum().item()

    n = len(loader.dataset)
    return total_loss / n, total_iou / n


# ── Visualisation ─────────────────────────────────────────────────────────────
def save_results(model, dataset, device, n=8, path="detector_results.png"):
    """Save a grid showing ground-truth (green) and predicted (red) boxes."""
    model.eval()
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 2, 8))
    axes = axes.flatten()

    indices = random.sample(range(len(dataset)), n)
    class_names = VOCDataset.CLASSES

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            img, cls_true, bbox_true = dataset[idx]
            cls_logits, bbox_pred = model(img.unsqueeze(0).to(device))
            cls_pred = cls_logits.argmax(dim=1).item()
            bbox_pred = bbox_pred.squeeze(0).cpu()

            # display image (H, W, C)
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis("off")

            scale = IMG_SIZE
            # ground truth box – green
            x0, y0 = bbox_true[0] * scale, bbox_true[1] * scale
            w,  h  = (bbox_true[2] - bbox_true[0]) * scale, (bbox_true[3] - bbox_true[1]) * scale
            ax.add_patch(patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor="lime", facecolor="none"))

            # predicted box – red
            px0, py0 = bbox_pred[0] * scale, bbox_pred[1] * scale
            pw,  ph  = (bbox_pred[2] - bbox_pred[0]) * scale, (bbox_pred[3] - bbox_pred[1]) * scale
            ax.add_patch(patches.Rectangle((px0, py0), pw, ph, linewidth=2, edgecolor="red", facecolor="none"))

            iou = compute_iou(bbox_pred.unsqueeze(0), bbox_true.unsqueeze(0)).item()
            ax.set_title(
                f"GT:{class_names[cls_true]}  Pred:{class_names[cls_pred]}\nIoU:{iou:.2f}",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    print(f"Results saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset — VOC 2012 provides official train / val splits
    train_set = VOCDataset(image_set='train')
    val_set   = VOCDataset(image_set='val')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model     = DetectorCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train IoU':>9}  {'Val Loss':>8}  {'Val IoU':>7}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, device)
        val_loss,   val_iou   = eval_epoch(model, val_loader, device)
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_iou:>9.4f}  {val_loss:>8.4f}  {val_iou:>7.4f}")

    torch.save(model.state_dict(), "detector_model.pth")
    print("\nModel saved to detector_model.pth")

    save_results(model, val_set, device)
