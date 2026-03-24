import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE     = 64
NUM_CLASSES  = 2          # 0 = circle, 1 = rectangle
DATASET_SIZE = 4000
BATCH_SIZE   = 64
EPOCHS       = 20
LR           = 1e-3

S            = 4     # grid size S×S (each cell = 16×16 px)
B            = 2     # bounding boxes predicted per grid cell
C            = NUM_CLASSES

LAMBDA_COORD = 5.0   # weight on localisation loss
LAMBDA_NOOBJ = 0.5   # weight on no-object confidence loss


# ── Dataset ──────────────────────────────────────────────────────────────────
class ShapeDataset(Dataset):
    """
    Identical to detect.py: 64×64 RGB images with one shape each.
    Returns (img_tensor, cls, bbox) where bbox = [x_min, y_min, x_max, y_max]
    normalised to [0, 1].
    """

    def __init__(self, size=DATASET_SIZE):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(20, 20, 20))
        draw = ImageDraw.Draw(img)

        shape_size = random.randint(10, 30)
        x0 = random.randint(0, IMG_SIZE - shape_size - 1)
        y0 = random.randint(0, IMG_SIZE - shape_size - 1)
        x1 = x0 + shape_size
        y1 = y0 + shape_size

        color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )

        cls = random.randint(0, 1)
        if cls == 0:
            draw.ellipse([x0, y0, x1, y1], fill=color)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=color)

        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        bbox = torch.tensor(
            [x0 / IMG_SIZE, y0 / IMG_SIZE, x1 / IMG_SIZE, y1 / IMG_SIZE],
            dtype=torch.float32,
        )
        return img_tensor, cls, bbox


# ── YOLO target builder ───────────────────────────────────────────────────────
def build_yolo_target(cls_batch, bbox_batch, s=S, b=B, c=C):
    """
    Convert (cls, bbox[x_min,y_min,x_max,y_max]) → YOLO grid target tensor.

    Returns: (N, S, S, B*5 + C)

    Encoding for the cell that owns the object centre:
      [x_cell, y_cell, w_img, h_img, conf] × B,  then one-hot class × C
      x_cell, y_cell : object centre offset within the cell, in [0, 1]
      w_img, h_img   : box width/height relative to the full image, in [0, 1]
      conf           : 1.0  (object present)
    All other cells are zero.
    """
    n      = cls_batch.size(0)
    target = torch.zeros(n, s, s, b * 5 + c, dtype=torch.float32)

    for i in range(n):
        x_min, y_min, x_max, y_max = bbox_batch[i]
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w  = x_max - x_min
        h  = y_max - y_min

        col = min(int(cx * s), s - 1)   # grid column (x-axis)
        row = min(int(cy * s), s - 1)   # grid row    (y-axis)

        x_cell = cx * s - col           # offset within cell
        y_cell = cy * s - row

        for bi in range(b):
            base = bi * 5
            target[i, row, col, base]     = x_cell
            target[i, row, col, base + 1] = y_cell
            target[i, row, col, base + 2] = w
            target[i, row, col, base + 3] = h
            target[i, row, col, base + 4] = 1.0

        target[i, row, col, b * 5 + cls_batch[i].item()] = 1.0

    return target


# ── Model ─────────────────────────────────────────────────────────────────────
class YOLOv1(nn.Module):
    """
    YOLOv1-style detector adapted for 64×64 input.

    Backbone: same 3-block CNN as DetectorCNN in detect.py
              (ReLU replaced with LeakyReLU as in the original paper).
    Head:     two FC layers that reshape to (S, S, B*5 + C).

    Per-cell prediction encoding:
      [x, y, w, h, conf] × B boxes, then C class logits.
      x, y are activated with sigmoid inside the loss/decode functions.
      w, h are left unconstrained (raw network values).
    """

    def __init__(self, s=S, b=B, c=C):
        super().__init__()
        self.s = s
        self.b = b
        self.c = c

        # Input: (N, 3, 64, 64)
        # After block 1: (N,  32, 32, 32)
        # After block 2: (N,  64, 16, 16)
        # After block 3: (N, 128,  8,  8)  → flatten → 8192
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(1024, s * s * (b * 5 + c)),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        out  = self.fc(feat)
        return out.view(-1, self.s, self.s, self.b * 5 + self.c)


# ── IoU utility ───────────────────────────────────────────────────────────────
def compute_iou(box_a, box_b):
    """
    IoU for boxes in [x_min, y_min, x_max, y_max] format.
    Supports arbitrary leading batch dimensions.
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
    union  = area_a + area_b - inter_area

    return inter_area / union.clamp(min=1e-6)


# ── NMS utility ───────────────────────────────────────────────────────────────
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression: removes redundant overlapping boxes.

    Args:
        boxes:         (N, 4) [x_min, y_min, x_max, y_max]
        scores:        (N,)   confidence score for each box
        iou_threshold: boxes with IoU > this value are suppressed

    Returns:
        kept: list of indices of boxes to keep
    """
    order = scores.argsort(descending=True)
    kept  = []

    while order.numel() > 0:
        best = order[0].item()
        kept.append(best)

        if order.numel() == 1:
            break

        rest  = order[1:]
        iou   = compute_iou(boxes[best].unsqueeze(0), boxes[rest])
        order = rest[iou <= iou_threshold]

    return kept


# ── YOLOv1 Loss ───────────────────────────────────────────────────────────────
def yolo_loss(predictions, targets, s=S, b=B, c=C,
              lambda_coord=LAMBDA_COORD, lambda_noobj=LAMBDA_NOOBJ):
    """
    YOLOv1 multi-part loss (per the original paper).

    predictions : (N, S, S, B*5 + C)  — raw network output
    targets     : (N, S, S, B*5 + C)  — from build_yolo_target()

    Components:
      1. λ_coord × xy loss           (responsible box, obj cells)
      2. λ_coord × sqrt(wh) loss     (responsible box, obj cells)
      3.           conf loss obj      (responsible box, obj cells; target = IOU)
      4. λ_noobj × conf loss noobj   (all boxes in no-obj cells +
                                      non-responsible boxes in obj cells)
      5.           class loss         (obj cells, MSE as in original paper)
    """
    n      = predictions.size(0)
    device = predictions.device

    # Apply sigmoid to x,y offsets and confidence; leave w,h raw
    pred = predictions.clone()
    for bi in range(b):
        base = bi * 5
        pred[..., base:base + 2] = torch.sigmoid(predictions[..., base:base + 2])
        pred[..., base + 4]      = torch.sigmoid(predictions[..., base + 4])

    obj_mask   = targets[..., 4]          # (N, S, S) — 1 where object exists
    noobj_mask = 1.0 - obj_mask

    tx = targets[..., 0]                  # ground-truth cell offsets / dims
    ty = targets[..., 1]
    tw = targets[..., 2]
    th = targets[..., 3]

    # Grid index tensors for converting cell-relative → absolute coords
    cell_x = torch.arange(s, dtype=torch.float32, device=device).view(1, 1, s).expand(n, s, s)
    cell_y = torch.arange(s, dtype=torch.float32, device=device).view(1, s, 1).expand(n, s, s)

    # Ground-truth box in absolute xyxy (normalised to image)
    tgt_cx   = (tx + cell_x) / s
    tgt_cy   = (ty + cell_y) / s
    tgt_xyxy = torch.stack([tgt_cx - tw / 2, tgt_cy - th / 2,
                             tgt_cx + tw / 2, tgt_cy + th / 2], dim=-1)

    # IOU between each predicted box and the ground-truth box
    ious = []
    for bi in range(b):
        base = bi * 5
        px   = pred[..., base];     py = pred[..., base + 1]
        pw   = pred[..., base + 2]; ph = pred[..., base + 3]
        pcx  = (px + cell_x) / s;  pcy = (py + cell_y) / s
        pb   = torch.stack([pcx - pw / 2, pcy - ph / 2,
                             pcx + pw / 2, pcy + ph / 2], dim=-1)
        ious.append(compute_iou(pb, tgt_xyxy))   # (N, S, S)

    # Responsible box = the one with highest IOU per cell
    _, best_b = torch.stack(ious, dim=-1).max(dim=-1)   # (N, S, S)

    total = torch.zeros(1, device=device)

    for bi in range(b):
        base  = bi * 5
        px    = pred[..., base];     py    = pred[..., base + 1]
        pw    = pred[..., base + 2]; ph    = pred[..., base + 3]
        pconf = pred[..., base + 4]

        resp     = (best_b == bi).float() * obj_mask     # responsible box mask
        non_resp = (1.0 - (best_b == bi).float()) * obj_mask

        # 1. xy loss
        total += lambda_coord * (
            (resp * (px - tx).pow(2)).sum() +
            (resp * (py - ty).pow(2)).sum()
        )

        # 2. sqrt(wh) loss — sign-aware sqrt handles negative raw w/h values
        sqrt_pw = pw.sign() * pw.abs().clamp(min=1e-6).sqrt()
        sqrt_ph = ph.sign() * ph.abs().clamp(min=1e-6).sqrt()
        sqrt_tw = tw.clamp(min=0).sqrt()
        sqrt_th = th.clamp(min=0).sqrt()
        total += lambda_coord * (
            (resp * (sqrt_pw - sqrt_tw).pow(2)).sum() +
            (resp * (sqrt_ph - sqrt_th).pow(2)).sum()
        )

        # 3. Confidence loss — responsible box in obj cells (target = IOU)
        total += (resp * (pconf - ious[bi].detach()).pow(2)).sum()

        # 4. Confidence loss — no-obj cells + non-responsible boxes in obj cells
        total += lambda_noobj * (
            (noobj_mask * pconf.pow(2)).sum() +
            (non_resp   * pconf.pow(2)).sum()
        )

    # 5. Class loss — MSE over class logits in obj cells (as in original paper)
    pred_cls = pred[..., b * 5:]
    tgt_cls  = targets[..., b * 5:]
    total += (obj_mask.unsqueeze(-1) * (pred_cls - tgt_cls).pow(2)).sum()

    return total / n


# ── Decode predictions ────────────────────────────────────────────────────────
def decode_best_box(pred, s=S, b=B, c=C):
    """
    Single-object decoding: return the box and class with the highest
    object confidence across all cells and box slots.

    pred    : (N, S, S, B*5 + C)
    Returns : boxes (N, 4) in [x_min, y_min, x_max, y_max], classes (N,)
    """
    n      = pred.size(0)
    device = pred.device

    pred_proc = pred.clone()
    for bi in range(b):
        base = bi * 5
        pred_proc[..., base:base + 2] = torch.sigmoid(pred[..., base:base + 2])
        pred_proc[..., base + 4]      = torch.sigmoid(pred[..., base + 4])

    cell_x = torch.arange(s, dtype=torch.float32, device=device).view(1, 1, s).expand(n, s, s)
    cell_y = torch.arange(s, dtype=torch.float32, device=device).view(1, s, 1).expand(n, s, s)

    best_boxes   = []
    best_classes = []

    for i in range(n):
        best_conf = -1.0
        best_box  = torch.tensor([0.0, 0.0, 0.1, 0.1])
        best_cls  = 0

        for bi in range(b):
            base  = bi * 5
            px    = pred_proc[i, :, :, base]
            py    = pred_proc[i, :, :, base + 1]
            pw    = pred_proc[i, :, :, base + 2].abs()
            ph    = pred_proc[i, :, :, base + 3].abs()
            pconf = pred_proc[i, :, :, base + 4]

            cls_ids = pred_proc[i, :, :, b * 5:].argmax(dim=-1)   # (S, S)

            flat_idx       = pconf.flatten().argmax().item()
            row, col       = divmod(flat_idx, s)
            conf           = pconf[row, col].item()

            if conf > best_conf:
                best_conf = conf
                cx = (px[row, col].item() + col) / s
                cy = (py[row, col].item() + row) / s
                w  = pw[row, col].item()
                h  = ph[row, col].item()
                best_box = torch.tensor([cx - w / 2, cy - h / 2,
                                         cx + w / 2, cy + h / 2])
                best_cls = cls_ids[row, col].item()

        best_boxes.append(best_box)
        best_classes.append(best_cls)

    return torch.stack(best_boxes), torch.tensor(best_classes)


# ── Training / evaluation loops ───────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_iou  = 0.0

    for imgs, cls_targets, bbox_targets in loader:
        imgs         = imgs.to(device)
        cls_targets  = cls_targets.to(device)
        bbox_targets = bbox_targets.to(device)

        yolo_targets = build_yolo_target(cls_targets, bbox_targets).to(device)

        optimizer.zero_grad()
        pred = model(imgs)
        loss = yolo_loss(pred, yolo_targets)
        loss.backward()
        optimizer.step()

        boxes, _ = decode_best_box(pred.detach())
        total_iou  += compute_iou(boxes.to(device), bbox_targets).sum().item()
        total_loss += loss.item() * imgs.size(0)

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

            yolo_targets = build_yolo_target(cls_targets, bbox_targets).to(device)

            pred = model(imgs)
            loss = yolo_loss(pred, yolo_targets)

            boxes, _ = decode_best_box(pred)
            total_iou  += compute_iou(boxes.to(device), bbox_targets).sum().item()
            total_loss += loss.item() * imgs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_iou / n


# ── Visualisation ─────────────────────────────────────────────────────────────
def save_results(model, dataset, device, n=8, path="yolo_results.png"):
    """Save a grid showing ground-truth (green) and predicted (red) boxes."""
    model.eval()
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 2, 8))
    axes = axes.flatten()

    indices     = random.sample(range(len(dataset)), n)
    class_names = ["circle", "rect"]

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            img, cls_true, bbox_true = dataset[idx]
            pred              = model(img.unsqueeze(0).to(device))
            boxes, classes    = decode_best_box(pred)
            bbox_pred         = boxes[0].cpu()
            cls_pred          = classes[0].item()

            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis("off")

            scale = IMG_SIZE

            # ground-truth box – green
            x0 = bbox_true[0] * scale;  y0 = bbox_true[1] * scale
            w  = (bbox_true[2] - bbox_true[0]) * scale
            h  = (bbox_true[3] - bbox_true[1]) * scale
            ax.add_patch(patches.Rectangle((x0, y0), w, h,
                                           linewidth=2, edgecolor="lime", facecolor="none"))

            # predicted box – red
            px0 = bbox_pred[0] * scale;  py0 = bbox_pred[1] * scale
            pw  = (bbox_pred[2] - bbox_pred[0]) * scale
            ph  = (bbox_pred[3] - bbox_pred[1]) * scale
            ax.add_patch(patches.Rectangle((px0, py0), pw, ph,
                                           linewidth=2, edgecolor="red", facecolor="none"))

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

    dataset    = ShapeDataset(size=DATASET_SIZE)
    val_size   = DATASET_SIZE // 5
    train_size = DATASET_SIZE - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = YOLOv1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train IoU':>9}  {'Val Loss':>8}  {'Val IoU':>7}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, device)
        val_loss,   val_iou   = eval_epoch(model, val_loader, device)
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_iou:>9.4f}  {val_loss:>8.4f}  {val_iou:>7.4f}")

    torch.save(model.state_dict(), "yolo_model.pth")
    print("\nModel saved to yolo_model.pth")

    save_results(model, val_set.dataset, device)
