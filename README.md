# lx-ai

A PyTorch-based training framework for multi-label classification of gastroenterology endoscopic images. Designed for both research and production workflows, with strong emphasis on data integrity, reproducibility, and flexible data sourcing.

The framework provides an end-to-end pipeline that handles:

- Dataset construction
- Label processing
- Split generation
- Model training
- Evaluation
- Reporting

All in a consistent and validated manner.

## Core Capabilities

### Multi-label Classification
- Supports multiple simultaneous labels per image (e.g., polyp, blood, instrument, etc.)
- Handles incomplete annotations using masking

### Flexible Data Sources
- **PostgreSQL database** — production/service mode
- **SQLite** — local development mode

### Robust Dataset Handling
- Label filtering by labelset version
- Explicit handling of:
  - Known positives
  - Known negatives
  - Unknown labels
- Configurable semantics:
  - Treat unknown as negative (closed-world)
  - Ignore unknown (open-world)

### Stable and Reproducible Data Splitting
- Bucket-based splitting with deterministic hashing
- Grouping by each video.
- Persistent video bucket registry to ensure:
  - No data leakage
  - Stable splits across runs
  - Reproducibility in experiments

### Model Architecture Flexibility
- GastroNet ResNet50 (recommended for medical domain)
- Standard backbones (ImageNet pretrained or random)
- Easy extension for new architectures

### Training Features
- Focal loss with:
  - Per-label class weighting
  - Masking for unknown labels
- Separate learning rates for backbone and head
- Optional backbone freezing
- Cosine annealing scheduler with warmup

### Evaluation and Metrics
- Global metrics: precision, recall, F1-score, accuracy
- Per-label metrics
- Support for:
  - Standard evaluation (with negatives)
  - Positives-only evaluation (when negatives are unavailable)

### Data Validation and Diagnostics
- Automatic dataset validation reports
- Label distribution analysis
- Split integrity checks
- Dataset imbalance detection
- Video and dataset-level diagnostics

### Reproducibility and Traceability
- Full configuration captured in metadata
- Saved model weights and training history
- Persistent bucket assignments
- Deterministic dataset splits

### Production and Development Compatibility
- Service mode using PostgreSQL (production)
- Local mode using SQLite (development)
- Frame path remapping for local debugging of production data

### Testing and Reliability
- Extensive unit test coverage for:
  - Configuration validation
  - Dataset building
  - Splitting logic
  - Bucket hashing and allocation
  - Loss functions and metrics
  - Database loaders

## Design Principles

lx-ai is built around a few key principles:

### No Data Leakage
Group-based splitting ensures frames from the same examination or video never cross splits.

### Reproducibility First
Persistent bucket assignment guarantees identical splits across runs and environments.

### Explicit Label Semantics
Unknown labels are never silently treated as negatives unless explicitly configured.

### Separation of Concerns
Data loading, splitting, training, and evaluation are modular and independently testable.

### Production-Aware Design
The same pipeline works in both local development and service-based production environments.

## Quick Start


### Paths a new developer should configure

Main place:

```bash
.env
```

Recommended local values: (write complete in all variables e.g DATA_DIR=/home/admin/dev/lx-ai/data not DATA_DIR=${WORKING_DIR}/data in .env file)

```bash
# Runtime roots
# Root of your system (user-specific)
HOME_DIR=/home/<your-user>
# Path where lx-ai repository is cloned
WORKING_DIR=${HOME_DIR}/dev/lx-ai
# Main data directory (all runtime data)
DATA_DIR=${WORKING_DIR}/data
# Configuration directory (passwords, configs)
CONF_DIR=${WORKING_DIR}/conf
# Storage root (usually same as DATA_DIR)
STORAGE_DIR=${DATA_DIR}
# Frame storage (extracted images)
FRAME_DIR=${DATA_DIR}/frames

# Training outputs
# Root for all training artifacts
TRAINING_ROOT=${DATA_DIR}/model_training
# Pretrained and saved model checkpoints
CHECKPOINTS_DIR=${TRAINING_ROOT}/checkpoints
# Training outputs (models, logs, metadata)
RUNS_DIR=${TRAINING_ROOT}/runs
# Bucket snapshots (split reproducibility)
BUCKET_SNAPSHOT_DIR=${TRAINING_ROOT}/buckets

# Model checkpoint
BACKBONE_CHECKPOINT=${CHECKPOINTS_DIR}/{model_weights}
# Training config-a relative path inside the repository.
TRAINING_CONFIG_PATH=lx_ai/ai_model_config/train_sandbox_postgres.yaml

# Optional CSV import - used by lx_ai/scripts/import_csv_sqlite.py
CSV_DIR=${DATA_DIR}/import/csv

# Local SQLite
SQLITE_DB_PATH=${WORKING_DIR}/database

```

### Database variables

#### Local development with SQLite

```bash
DB_BACKEND=
DJANGO_SETTINGS_MODULE=
DJANGO_DB_ENGINE=
SQLITE_DB_PATH=
```

For SQLite, these PostgreSQL-style values may exist but are not the active DB connection:

```bash
DJANGO_DB_HOST=
DJANGO_DB_PORT=
DJANGO_DB_NAME=
DJANGO_DB_USER=
```

#### Production or service with PostgreSQL

These are normally generated in `.env.systemd` by the Luxnix service:

```bash
DB_BACKEND=
DJANGO_SETTINGS_MODULE=SQLITE_DB_PATH=${WORKING_DIR}.settings.settings_prod
DJANGO_DB_ENGINE=
DJANGO_DB_HOST=localhost
DJANGO_DB_PORT=
DJANGO_DB_NAME=<database_name>
DJANGO_DB_USER=<database_user>
DJANGO_DB_PASSWORD_FILE=${CONF_DIR}/db_pwd  # e.g /var/endoreg-service-user/lx-ai/conf/db_pwd
DJANGO_DB_SSLMODE=prefer
```

### Files where these are used

| Path variable             | Purpose                                 | Used in                                        |
|---------------------------|---------|---------------------------------------------|
| `DATA_DIR`                | Main data root                          | `secretspec.toml`, `devenv.nix`, training YAML |
| `CONF_DIR`                | Password and config files               | `secretspec.toml`, service `.env.systemd`      |
| `FRAME_DIR`               | Default frame directory                 | Django settings and diagnostics                |
| `TRAINING_ROOT`           | Training artifact root                  | training config                                |
| `CHECKPOINTS_DIR`         | Backbone checkpoint folder              | training config                                |
| `RUNS_DIR`                | Saved models, metadata, reports         | training config                                |
| `BUCKET_SNAPSHOT_DIR`     | Bucket snapshots                        | `lx_ai/training/bucket_snapshot.py`            |
| `BACKBONE_CHECKPOINT`     | GastroNet checkpoint path               | `train_sandbox_postgres.yaml`                  |
| `TRAINING_CONFIG_PATH`    | Which YAML file `run_training.py` loads | `lx_ai/run_training.py`                        |
| `LEGACY_IMAGE_DIR`        | JSONL image folder                      | `data_loader_for_model_input.py`               |
| `LEGACY_JSONL_PATH`       | JSONL annotation file                   | `data_loader_for_model_input.py`               |
| `CSV_DIR`                 | CSV import folder                       | `scripts/import_csv_sqlite.py`                 |
| `SQLITE_DB_PATH`          | Local SQLite DB file                    | SQLite loaders                                 |
| `FRAME_PATH_REMAP_SOURCE` | Original service frame path prefix      | `data_loader_for_model_training.py`            |
| `FRAME_PATH_REMAP_TARGET` | Local mirrored frame path prefix        | `data_loader_for_model_training.py`            |

### Training configuration

Edit the following file:

```bash
lx_ai/ai_model_config/train_sandbox_postgres.yaml
```

Important fields to configure:

```yaml
dataset_ids: [1, 2]
labelset_id: 5
labelset_version_to_train: 3

backbone_name:
backbone_checkpoint: "$BACKBONE_CHECKPOINT"

base_dir: "$DATA_DIR"
training_root: "$TRAINING_ROOT"
checkpoints_dir: "$CHECKPOINTS_DIR"
runs_dir: "$RUNS_DIR"
```

**For a new dataset**, update:

```yaml
dataset_ids: [1, 2, 3]
```

**For a different labelset**, update:

```yaml
labelset_id: <your_labelset_id>
labelset_version_to_train: <your_version>
```

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
cd /lx-ai
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
backbone_checkpoint: /path/to/model_weights.pth
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
export FRAME_PATH_REMAP_SOURCE=""
export FRAME_PATH_REMAP_TARGET=""
```

### Missing GastroNet checkpoint

Verify the path for `model_weights.pth`.

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
backbone_checkpoint: /path/to/model_weights.pth
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
export FRAME_PATH_REMAP_SOURCE="/"
export FRAME_PATH_REMAP_TARGET="/home/admin/dev/lx-ai/data/frames_mirror"
```

### Missing GastroNet checkpoint

Verify the path for `model_weights.pth`.

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
