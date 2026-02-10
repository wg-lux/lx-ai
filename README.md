# lx-ai

A PyTorch-based multi-label image classification training framework for endoscopic image analysis.

## Overview

**lx-ai** is a machine learning pipeline for training multi-label classification models on gastroenterology endoscopic images. It supports:

- **Multi-label classification** of medical images (polyp, blood, water jet, etc.)
- **Flexible data sources** (PostgreSQL database or legacy JSONL + image directories)
- **GastroNet backbone** with ResNet50 feature extraction
- **Focal loss** with per-label class weighting and masking for unknown labels
- **Group-wise dataset splitting** by examination ID for robust train/val/test separation
- **Comprehensive training logging** and per-label metrics

## Features

### Data Loading
- Load annotations directly from PostgreSQL (endoreg-db compatible)
- Support for legacy JSONL format with image directories
- Automatic label filtering by labelset version
- Configurable handling of unlabeled data (treat as negative or ignore)

### Model Architecture
- **Backbone options:**
  - `gastro_rn50` — ResNet50 with GastroNet checkpoint (recommended)
  - `resnet50_imagenet` — ImageNet pretrained ResNet50
  - `resnet50_random` — Random weight ResNet50
  - `efficientnet_b0_imagenet` — ImageNet pretrained EfficientNet-B0

- **Training head:** Fully connected layer for multi-label logits
- **Freezable backbone:** Option to freeze backbone weights for transfer learning

### Training Features
- **Focal loss** with configurable α (balance) and γ (focus) parameters
- **Per-label class weights** computed from positive sample counts
- **Learning rate scheduling:** Cosine annealing with optional warmup
- **Separate learning rates** for backbone and classification head
- **Validation & test splits** with group-wise stratification by examination ID

### Metrics & Logging
- Global metrics: precision, recall, F1-score, accuracy, confusion counts
- Per-label metrics for detailed analysis
- Structured console output with color support
- Automatic model + metadata saving

## Project Structure

```
lx-ai/
├── lx_ai/
│   ├── ai_model/                          # Model architecture
│   │   ├── model_backbones.py            # Backbone factory (ResNet, EfficientNet)
│   │   ├── model_gastronet_resnet.py     # GastroNet wrapper
│   │   ├── losses.py                     # Focal loss + class weights
│   │   └── ...
│   ├── ai_model_dataset/
│   │   └── dataset.py                    # PyTorch Dataset + spec validation
│   ├── ai_model_matrics/
│   │   └── metrics.py                    # Precision, recall, F1, per-label stats
│   ├── ai_model_training/
│   │   └── trainer_gastronet_multilabel.py  # Main training loop
│   ├── ai_model_config/
│   │   ├── config.py                     # TrainingConfig (Pydantic model)
│   │   └── train_sandbox_postgres.yaml   # Example config (PostgreSQL)
│   ├── utils/
│   │   ├── data_loader_for_model_input.py   # Dataset building (JSONL + DB)
│   │   ├── db_loader_for_model_input.py     # PostgreSQL loaders
│   │   ├── logging_utils.py                 # Console formatting
│   │   └── ...
│   └── run_training.py                   # Main entry point
├── data/
│   ├── model_training/
│   │   ├── checkpoints/                  # Pretrained backbone weights
│   │   └── runs/                         # Trained model outputs
│   └── ...
├── tests/                                # Unit tests
├── pyproject.toml                        # Python project config
├── .env.sandbox                          # Sandbox database credentials
├── .env.sandbox.example                  # Template for environment variables
└── README.md                             # This file
```

## Installation

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support (optional but recommended)
- PostgreSQL (for database mode)

### Setup

1. **Clone and environment setup:**
   ```bash
   git clone <repo-url>
   cd lx-ai
   uv sync
   direnv allow
   source /home/admin/dev/lx-ai/.devenv/state/venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -e libs/lx-data-models  # if using lx_dtypes from local lib
   set -a && source .env.sandbox && set +a
   unset DEV_DB_PASSWORD  # Use password file instead
   ```

3. **Verify setup:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Configuration

### Training Config File

Create or edit a YAML config file (e.g., `lx_ai/ai_model_config/train_sandbox_postgres.yaml`):

```yaml
# Dataset selection
dataset_uuid: sandbox_ds
data_source: postgres              # or "jsonl"
dataset_id: 1
labelset_id: 2
labelset_version_to_train: 2

# Data semantics
treat_unlabeled_as_negative: true  # false = ignore unknowns

# Model
backbone_name: gastro_rn50
backbone_checkpoint: /path/to/RN50_GastroNet-1M_DINOv1.pth
freeze_backbone: true

# Training hyperparameters
num_epochs: 20
batch_size: 16
val_split: 0.1
test_split: 0.1

# Learning rates
lr_head: 0.001
lr_backbone: 0.0001

# Focal loss
gamma_focal: 2.0
alpha_focal: 0.25

# Scheduler
use_scheduler: true
warmup_epochs: 2
min_lr: 1.0e-6

# Hardware
device: cuda
random_seed: 42
```

### Environment Variables

Set up `.env.sandbox` with database credentials:

```bash
DEV_DB_HOST=127.0.0.1
DEV_DB_PORT=15432
DEV_DB_NAME=endoreg_sandbox
DEV_DB_USER=endoreg_sandbox_user
DEV_DB_PASSWORD_FILE=/etc/secrets/vault/SCRT_local_password_maintenance_password
```

Or use direct password (less secure):
```bash
DEV_DB_PASSWORD=your_password
```

## Usage

### Running Training Pipeline

```bash
# Activate environment
set -a && source .env.sandbox && set +a
unset DEV_DB_PASSWORD

# Run training
python -m lx_ai.run_training
```

### Remote Database Access (PostgreSQL)

If your database is on a remote machine:

```bash
# Terminal 1: SSH tunnel (keep open)
ssh -N -L 15432:127.0.0.1:5432 admin@gs-02

# Terminal 2: Connect to database
psql "host=127.0.0.1 port=15432 dbname=endoreg_sandbox user=endoreg_sandbox_user"
```

### Data Sources

#### PostgreSQL Mode
```python
config = TrainingConfig.from_yaml_file(
    Path("lx_ai/ai_model_config/train_sandbox_postgres.yaml")
)
# Automatically loads:
# - annotations from endoreg_db_aidataset_image_annotations
# - labelset from endoreg_db_labelset + labels
```

#### JSONL Mode
```python
config = TrainingConfig(
    data_source="jsonl",
    jsonl_path=Path("data/legacy_images/legacy_img_dicts.jsonl"),
    treat_unlabeled_as_negative=True,
)
```

JSONL format (one JSON object per line):
```json
{"labels": ["polyp", "blood"], "old_examination_id": 25, "old_id": 479228, "filename": "479228.jpg"}
```

## Training Pipeline

### 1. Data Loading
- Fetch annotations + images (PostgreSQL or JSONL)
- Validate dataset structure
- Filter labels by labelset version
- Drop labels with zero positive samples

### 2. Dataset Split
- **Group-wise split** by `old_examination_id`
- Ensures frames from same exam don't leak between train/val/test
- Stratified sampling: shuffle groups, then assign by ratio

### 3. Model Initialization
- Load backbone (ResNet50 + optional GastroNet checkpoint)
- Attach linear classification head (num_labels outputs)
- Optionally freeze backbone weights
- Move model to device (CUDA/CPU)

### 4. Training Loop (per epoch)
- **Forward pass:** images → backbone features → logits
- **Compute loss:** focal loss with per-label class weights + masking
- **Backward:** gradient descent on head + optionally backbone
- **LR scheduling:** cosine annealing with warmup (optional)

### 5. Validation
- Evaluate on validation split
- Compute global + per-label metrics
- Log results

### 6. Testing
- Final evaluation on test split
- Save model weights + metadata JSON

## Outputs

After training, check `data/model_training/runs/`:

```
dataset_{uuid}_{backbone}_v{version}_multilabel.pth         # Model weights
dataset_{uuid}_{backbone}_v{version}_multilabel_meta.json   # Metadata
```

### Metadata JSON Structure
```json
{
  "config": { ... },                    # Full TrainingConfig as dict
  "labelset": { "id": 2, "version": 2, "labels": [...] },
  "used_label_names": ["polyp", "blood", ...],
  "used_label_indices_original": [0, 1, ...],
  "history": {
    "train_loss": [...],
    "val_loss": [...]
  },
  "test_metrics_final": {
    "precision": 0.85,
    "recall": 0.82,
    "f1": 0.83,
    "accuracy": 0.88,
    "tp": 1234, "fp": 200, "tn": 5000, "fn": 100,
    "per_label": [
      {"precision": 0.90, "recall": 0.88, "f1": 0.89, "support": 150},
      ...
    ]
  }
}
```

## Key Classes & Functions

### Configuration
- [`TrainingConfig`](lx_ai/ai_model_config/config.py) — Pydantic model for training hyperparameters
- Loads from YAML, validates paths, computes defaults

### Data
- [`MultiLabelDatasetSpec`](lx_ai/ai_model_dataset/dataset.py) — Validated dataset boundary contract
- [`EndoMultiLabelDataset`](lx_ai/ai_model_dataset/dataset.py) — PyTorch Dataset (images + labels + masks)
- [`build_dataset_for_training()`](lx_ai/utils/data_loader_for_model_input.py) — Unified loader (JSONL/DB)

### Model
- [`create_multilabel_model()`](lx_ai/ai_model/model_backbones.py) — Backbone + head factory
- [`MultiLabelBackboneHead`](lx_ai/ai_model/model_backbones.py) — Model wrapper (trainer interface)

### Training
- [`train_gastronet_multilabel()`](lx_ai/ai_model_training/trainer_gastronet_multilabel.py) — Main training loop
- [`focal_loss_with_mask()`](lx_ai/ai_model/losses.py) — Multi-label focal loss with class weights
- [`compute_metrics()`](lx_ai/ai_model_matrics/metrics.py) — Global + per-label metrics

## Advanced Usage

### Custom Backbone

```python
from lx_ai.ai_model.model_backbones import create_multilabel_model

model = create_multilabel_model(
    backbone_name="resnet50_imagenet",  # or other options
    num_labels=9,
    backbone_checkpoint=None,
    freeze_backbone=False,
)
```

### Custom Loss Configuration

```python
from lx_ai.ai_model.losses import FocalLossConfig, focal_loss_with_mask

cfg = FocalLossConfig(alpha=0.5, gamma=1.5)
loss = focal_loss_with_mask(
    logits=model_output,
    targets=labels,
    masks=known_masks,
    class_weights=weights,
    **cfg.model_dump(),
)
```

### Inference on New Images

```python
import torch
from PIL import Image
from pathlib import Path

# Load model
model = create_multilabel_model(
    backbone_name="gastro_rn50",
    num_labels=9,
    backbone_checkpoint=Path("data/model_training/runs/model.pth"),
    freeze_backbone=True,
)
model.eval()

# Preprocess image (same as training)
img = Image.open("sample.jpg").convert("RGB").resize((224, 224))
img_tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
img_tensor = (img_tensor - MEAN) / STD

# Inference
with torch.no_grad():
    logits = model(img_tensor.unsqueeze(0))
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()
```

## Troubleshooting

### Database Connection Issues
```bash
# Verify SSH tunnel is open
lsof -i :15432

# Test connection
psql "host=127.0.0.1 port=15432 dbname=endoreg_sandbox user=endoreg_sandbox_user"
```

### Environment Variable Issues
```bash
# Clear any conflicting variables
unset DJANGO_SETTINGS_MODULE DEV_DB_ENGINE

# Load sandbox env
set -a && source .env.sandbox && set +a

# Verify
echo $DEV_DB_HOST $DEV_DB_PORT
```

### CUDA/Device Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Force CPU (if CUDA issues)
# Edit config: device: cpu
```

### Model Not Loading
```bash
# Verify checkpoint file exists and is readable
ls -lh data/model_training/checkpoints/RN50_GastroNet-1M_DINOv1.pth

# Check for Pylance torchvision warnings (set up interpreter correctly in VSCode)
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
- Type hints: Pylance-compatible (avoid `# type: ignore` when possible)
- Validation: Pydantic for boundary contracts
- Logging: Use [`section()` / `subsection()`](lx_ai/utils/logging_utils.py) for structured output

### Adding New Backbones

1. Add factory function in [`model_backbones.py`](lx_ai/ai_model/model_backbones.py)
2. Update `BackboneName` literal type
3. Add case in `create_multilabel_model()`

## License

MIT License © 2025 AG-Lux

See [LICENSE](LICENSE) for details.

## Citation

If you use lx-ai in your research, please cite:

```bibtex
@software{lxai2025,
  title={lx-ai: Multi-Label Image Classification for Endoscopy},
  author={AG-Lux},
  year={2025},
  url={https://github.com/ag-lux/lx-ai}
}
```

## Support

For issues, questions, or contributions:
- Check [CONTRIBUTING.md](CONTRIBUTING.md)
- Open an issue on GitHub
- Contact: [your contact info]