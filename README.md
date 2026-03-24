# Computer Vision

A collection of PyTorch experiments covering progressively more complex deep learning tasks — from a simple MLP regressor to a YOLOv1-style object detector.

## Projects

### 1. Simple MLP Regressor (`train.py` / `infer.py`)

A two-hidden-layer fully connected network trained on a synthetic regression task.

- Input: 20-dimensional random vectors
- Target: linear function of the first feature with Gaussian noise
- Loss: MSELoss, optimizer: Adam

```bash
python train.py   # trains and saves model.pth
python infer.py   # loads model.pth and runs inference on 5 random samples
```

---

### 2. MNIST Digit Classifier (`train_mnist.py` / `test_mnist.py`)

A small CNN for handwritten digit classification on the MNIST dataset.

- Architecture: 2 conv blocks (32 → 64 channels) + FC classifier
- Loss: CrossEntropyLoss, optimizer: Adam
- Downloads MNIST automatically to `data/`

```bash
python train_mnist.py   # trains and saves mnist_model.pth
python test_mnist.py    # evaluates on 10 000 test samples, prints per-sample results
```

Also available in **C++ (LibTorch)**:

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make
./train_mnist   # trains and saves mnist_model_cpp.pt
./infer         # runs inference using the saved TorchScript model
```

---

### 3. Pascal VOC Object Detector (`detect.py`)

A dual-head CNN trained on Pascal VOC 2012 to simultaneously classify and localise objects.

- 20 VOC classes, images resized to 224x224
- Backbone: 4 conv blocks (3 → 32 → 64 → 128 → 256 channels)
- Heads: classification (CrossEntropyLoss) + bounding box regression (SmoothL1Loss)
- Metric: IoU between predicted and ground-truth boxes
- Includes IoU and NMS utilities

```bash
python detect.py   # downloads VOC 2012, trains, saves detector_model.pth
                   # and writes detector_results.png (GT vs predicted boxes)
```

---

### 4. YOLOv1-Style Detector (`yolo_detect.py`)

A from-scratch YOLOv1 implementation trained on a synthetic shapes dataset.

- Dataset: 4000 procedurally generated 64x64 RGB images, each containing one circle or rectangle
- Grid: 4x4, 2 boxes per cell, 2 classes
- Full YOLOv1 multi-part loss (coord, sqrt-wh, confidence, no-obj, class)
- Backbone uses LeakyReLU as in the original paper
- Includes grid target builder, box decoder, IoU, and NMS

```bash
python yolo_detect.py   # trains, saves yolo_model.pth, writes yolo_results.png
```

## Requirements

CUDA 13.0 is required for the prebuilt wheels in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

Key dependencies: `torch==2.10.0+cu130`, `torchvision==0.26.0`, `numpy`, `matplotlib`.

For the C++ targets, download LibTorch from pytorch.org and set `CMAKE_PREFIX_PATH` accordingly. The `libtorch/` directory in this repo is expected to contain the LibTorch distribution.

## Saved Artifacts

| File | Description |
|---|---|
| `model.pth` | Simple MLP regressor weights |
| `mnist_model.pth` | MNIST CNN weights (Python) |
| `mnist_model_cpp.pt` | MNIST TorchScript model (C++) |
| `detector_model.pth` | VOC detector weights |
| `yolo_model.pth` | YOLOv1 detector weights |
| `detector_results.png` | VOC detector visualisation |
| `yolo_results.png` | YOLO detector visualisation |
