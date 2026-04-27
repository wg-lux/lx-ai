# lx-ai

A PyTorch-based multi-label image classification training framework for endoscopic image analysis.

## Overview

`lx-ai` is a training pipeline for multi-label classification of gastroenterology endoscopic images. It supports:
# lx-ai

A PyTorch-based multi-label image classification training framework for endoscopic image analysis.

## Overview

`lx-ai` is a training pipeline for multi-label classification of gastroenterology endoscopic images. It supports:

- PostgreSQL and JSONL data sources
- GastroNet ResNet50 and standard backbone options
- Focal loss with per-label weighting and unknown label masking
- Stable train/validation/test split by video or examination grouping
- Model export with metadata and training history
- Unit tests for core config, dataset, metrics, loss, and split logic

## Branches and Database Usage

### `sandbox`

Used for sandbox database work and direct PostgreSQL access.

### `prototype`

Used for the service-compatible workflow and local development.

- Service mode: PostgreSQL
- Local mode: SQLite

## Quick Start

### Main entry point

```bash
python lx_ai/run_training.py
```

Or as a module:

```bash
python -m lx_ai.run_training
```

### Recommended development workflow

```bash
cd /home/admin/dev/lx-ai
devenv shell
python lx_ai/run_training.py
```

## Configuration

The primary training config file is:

```text
lx_ai/ai_model_config/train_sandbox_postgres.yaml
```

It controls:

- dataset and labelset selection
- data source
- model backbone and checkpoint
- training hyperparameters
- scheduler settings
- device selection
- unknown-label behavior

Example fields:

```yaml
dataset_uuid: sandbox_ds
data_source: postgres
dataset_ids: [1, 2]
labelset_id: 5
labelset_version_to_train: 3
treat_unlabeled_as_negative: false
backbone_name: gastro_rn50
backbone_checkpoint: /path/to/RN50_GastroNet-1M_DINOv1.pth
freeze_backbone: true
num_epochs: 20
batch_size: 16
lr_head: 0.001
lr_backbone: 0.0001
gamma_focal: 2.0
alpha_focal: 0.25
use_scheduler: true
warmup_epochs: 2
min_lr: 1.0e-6
device: cuda
random_seed: 42
bucket_policy:
  num_buckets: 5
  validation_buckets: [3]
  test_buckets: [4]
save_bucket_snapshot: false
```

## Supported Backbones

Supported backbone names:

- `gastro_rn50`
- `resnet50_imagenet`
- `resnet50_random`
- `efficientnet_b0_imagenet`

Backbones are implemented in `lx_ai/ai_model/model_backbones.py`.

## Data Sources

### PostgreSQL mode

Use `data_source: postgres`.

Database loaders are in:

- `lx_ai/utils/db_loader_for_model_input.py`
- `lx_ai/utils/data_loader_for_model_input.py`

Connection variables are resolved from:

- `DEV_DB_*` first
- `DJANGO_DB_*` second

Password resolution supports:

- `*_PASSWORD`
- `*_PASSWORD_FILE`

### SQLite mode

For local development, set:

```bash
export DB_BACKEND=sqlite
```

The local loader supports SQLite through the same input pipeline.

### JSONL mode

Use `data_source: jsonl` and provide:

```yaml
jsonl_path: /path/to/data.jsonl
```

Expected JSONL format:

```json
{"labels": ["polyp"], "old_examination_id": 1, "old_id": 10, "filename": "10.jpg"}
```

## Unknown Label Handling

Two modes are supported:

- `treat_unlabeled_as_negative: false`
  - Unknown labels are masked out
  - Loss and metrics ignore unknown values
- `treat_unlabeled_as_negative: true`
  - Unknown labels are treated as negative
  - Use only when missing labels imply negative examples

## Dataset and Bucket Splitting

The loader builds datasets with:

- frame-level multi-label vectors
- label masks for unknown annotations
- stable video/examination split assignment
- bucket policy support for train/validation/test

Bucket policy example:

```yaml
bucket_policy:
  num_buckets: 5
  validation_buckets: [3]
  test_buckets: [4]
```

Training buckets are all remaining buckets not assigned to validation or test.

The split logic preserves:

- same-video grouping
- stable bucket assignments
- split exclusivity
- dataset integrity

## Training Flow

Training is managed by:

- `lx_ai/ai_model_training/trainer_gastronet_multilabel.py`

Typical steps:

1. Load `TrainingConfig`
2. Build dataset
3. Validate labels and sources
4. Create PyTorch datasets and loaders
5. Create model and optimizer
6. Train for configured epochs
7. Validate and select best checkpoint
8. Evaluate test split
9. Save model weights and metadata

## Loss and Metrics

### Loss

Loss implementation:

- `lx_ai/ai_model/losses.py`

Uses:

- `focal_loss_with_mask`
- `compute_class_weights`

Supports:

- multi-label logits
- per-label weights
- label masks
- focal alpha and gamma

### Metrics

Metrics implementation:

- `lx_ai/ai_model_matrics/metrics.py`

Supported metrics:

- precision
- recall
- F1
- accuracy
- TP / FP / TN / FN
- per-label metrics
- positives-only metrics when negatives are unavailable

## Outputs

Trained model artifacts are saved under:

```text
data/model_training/runs/
```

Output files:

- `dataset_<dataset_uuid>_<backbone_name>_v<labelset_version>_multilabel.pth`
- `dataset_<dataset_uuid>_<backbone_name>_v<labelset_version>_multilabel_meta.json`

Metadata includes:

- config
- labelset
- used labels
- training history
- final test metrics
- bucket policy and sizes

## Troubleshooting

### Missing labelset

Verify `labelset_id` and `labelset_version_to_train` exist in the database.

### Empty dataset

Check `dataset_ids` and ensure annotations exist for those IDs.

### Image file not found

For local development against service database paths, remap frame roots:

```bash
export FRAME_PATH_REMAP_SOURCE="/var/endoreg-service-user/lx-annotate/data/frames"
export FRAME_PATH_REMAP_TARGET="/home/admin/dev/lx-ai/data/frames_mirror"
```

### Missing GastroNet checkpoint

Verify the path for `RN50_GastroNet-1M_DINOv1.pth`.

### PostgreSQL password errors

Use one of:

```bash
export DEV_DB_PASSWORD=your_password
```

or

```bash
export DEV_DB_PASSWORD_FILE=/path/to/password/file
```

Service mode uses the `DJANGO_DB_*` equivalents.

## Testing

Run tests with:

```bash
pytest -q
```

Run a single file:

```bash
pytest tests/ai_model_config/test_training_config.py -q --no-cov
```
Run both:

```bash
pytest -q
```

## Project Structure

```text
lx-ai/
├── lx_ai/
│   ├── ai_model/
│   ├── ai_model_config/
│   ├── ai_model_dataset/
│   ├── ai_model_matrics/
│   ├── ai_model_split/
│   ├── ai_model_training/
│   ├── data_validation/
│   ├── scripts/
│   ├── utils/
│   └── run_training.py
├── tests/
├── data/
├── pyproject.toml
└── README.md
```

## License

MIT License © 2025 AG-Lux

See `LICENSE` for details.
- PostgreSQL and JSONL data sources
- GastroNet ResNet50 and standard backbone options
- Focal loss with per-label weighting and unknown label masking
- Stable train/validation/test split by video or examination grouping
- Model export with metadata and training history
- Unit tests for core config, dataset, metrics, loss, and split logic

## Branches and Database Usage

### `gs02_sandbox_db`

Used for sandbox database work and direct PostgreSQL access on our own server gs-02.
how to run it can be found here ```https://github.com/wg-lux/lx-ai/wiki/AI-Model---Running-Commands#lx-ai---gs02_sandbox_db```

### `prototype`

Used for the service-compatible workflow and local development.

- Service mode: PostgreSQL
- Local mode: SQLite

## Quick Start

### Main entry point

```bash
python lx_ai/run_training.py
```

Or as a module:

```bash
python -m lx_ai.run_training
```

### Recommended development workflow

```bash
cd /home/admin/dev/lx-ai
devenv shell
python lx_ai/run_training.py
```

## Configuration

The primary training config file is:

```text
lx_ai/ai_model_config/train_sandbox_postgres.yaml
```

It controls:

- dataset and labelset selection
- data source
- model backbone and checkpoint
- training hyperparameters
- scheduler settings
- device selection
- unknown-label behavior

Example fields:

```yaml
dataset_uuid: sandbox_ds
data_source: postgres
dataset_ids: [1, 2]
labelset_id: 5
labelset_version_to_train: 3
treat_unlabeled_as_negative: false
backbone_name: gastro_rn50
backbone_checkpoint: /path/to/RN50_GastroNet-1M_DINOv1.pth
freeze_backbone: true
num_epochs: 20
batch_size: 16
lr_head: 0.001
lr_backbone: 0.0001
gamma_focal: 2.0
alpha_focal: 0.25
use_scheduler: true
warmup_epochs: 2
min_lr: 1.0e-6
device: cuda
random_seed: 42
bucket_policy:
  num_buckets: 5
  validation_buckets: [3]
  test_buckets: [4]
save_bucket_snapshot: false
```

## Supported Backbones

Supported backbone names:

- `gastro_rn50`
- `resnet50_imagenet`
- `resnet50_random`
- `efficientnet_b0_imagenet`

Backbones are implemented in `lx_ai/ai_model/model_backbones.py`.

## Data Sources

### PostgreSQL mode

Use `data_source: postgres`.

Database loaders are in:

- `lx_ai/utils/db_loader_for_model_input.py`
- `lx_ai/utils/data_loader_for_model_input.py`

Connection variables are resolved from:

- `DEV_DB_*` first
- `DJANGO_DB_*` second

Password resolution supports:

- `*_PASSWORD`
- `*_PASSWORD_FILE`

### SQLite mode

For local development, set:

```bash
export DB_BACKEND=sqlite
```

The local loader supports SQLite through the same input pipeline.

### JSONL mode

Use `data_source: jsonl` and provide:

```yaml
jsonl_path: /path/to/data.jsonl
```

Expected JSONL format:

```json
{"labels": ["polyp"], "old_examination_id": 1, "old_id": 10, "filename": "10.jpg"}
```

## Unknown Label Handling

Two modes are supported:

- `treat_unlabeled_as_negative: false`
  - Unknown labels are masked out
  - Loss and metrics ignore unknown values
- `treat_unlabeled_as_negative: true`
  - Unknown labels are treated as negative
  - Use only when missing labels imply negative examples

## Dataset and Bucket Splitting

The loader builds datasets with:

- frame-level multi-label vectors
- label masks for unknown annotations
- stable video/examination split assignment
- bucket policy support for train/validation/test

Bucket policy example:

```yaml
bucket_policy:
  num_buckets: 5
  validation_buckets: [3]
  test_buckets: [4]
```

Training buckets are all remaining buckets not assigned to validation or test.

The split logic preserves:

- same-video grouping
- stable bucket assignments
- split exclusivity
- dataset integrity

## Training Flow

Training is managed by:

- `lx_ai/ai_model_training/trainer_gastronet_multilabel.py`

Typical steps:

1. Load `TrainingConfig`
2. Build dataset
3. Validate labels and sources
4. Create PyTorch datasets and loaders
5. Create model and optimizer
6. Train for configured epochs
7. Validate and select best checkpoint
8. Evaluate test split
9. Save model weights and metadata

## Loss and Metrics

### Loss

Loss implementation:

- `lx_ai/ai_model/losses.py`

Uses:

- `focal_loss_with_mask`
- `compute_class_weights`

Supports:

- multi-label logits
- per-label weights
- label masks
- focal alpha and gamma

### Metrics

Metrics implementation:

- `lx_ai/ai_model_matrics/metrics.py`

Supported metrics:

- precision
- recall
- F1
- accuracy
- TP / FP / TN / FN
- per-label metrics
- positives-only metrics when negatives are unavailable

## Outputs

Trained model artifacts are saved under:

```text
data/model_training/runs/
```

Output files:

- `dataset_<dataset_uuid>_<backbone_name>_v<labelset_version>_multilabel.pth`
- `dataset_<dataset_uuid>_<backbone_name>_v<labelset_version>_multilabel_meta.json`

Metadata includes:

- config
- labelset
- used labels
- training history
- final test metrics
- bucket policy and sizes

## Troubleshooting

### Missing labelset

Verify `labelset_id` and `labelset_version_to_train` exist in the database.

### Empty dataset

Check `dataset_ids` and ensure annotations exist for those IDs.

### Image file not found

For local development against service database paths, remap frame roots:

```bash
export FRAME_PATH_REMAP_SOURCE="/var/endoreg-service-user/lx-annotate/data/frames"
export FRAME_PATH_REMAP_TARGET="/home/admin/dev/lx-ai/data/frames_mirror"
```

### Missing GastroNet checkpoint

Verify the path for `RN50_GastroNet-1M_DINOv1.pth`.

### PostgreSQL password errors

Use one of:

```bash
export DEV_DB_PASSWORD=your_password
```

or

```bash
export DEV_DB_PASSWORD_FILE=/path/to/password/file
```

Service mode uses the `DJANGO_DB_*` equivalents.

## Testing

Run tests with:

```bash
pytest -q
```

and

```bash
pytest --cov=lx_ai --cov-report=html

# then open
firefox htmlcov/index.html 
```

Run a single file:

```bash
pytest tests/ai_model_config/test_training_config.py -q --no-cov
```

Run both:

```bash
pytest -q
```

## Project Structure

```text
lx-ai/
├── lx_ai/
│   ├── ai_model/
│   ├── ai_model_config/
│   ├── ai_model_dataset/
│   ├── ai_model_matrics/
│   ├── ai_model_split/
│   ├── ai_model_training/
│   ├── data_validation/
│   ├── scripts/
│   ├── utils/
│   └── run_training.py
├── tests/
├── data/
├── pyproject.toml
└── README.md
```

## License

MIT License © 2025 AG-Lux

See `LICENSE` for details.