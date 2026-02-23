# lx_ai/ai_model_config/config.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Literal, Optional, List, TypedDict, cast

from pydantic import AwareDatetime, ConfigDict, Field, field_serializer, field_validator, model_validator
from lx_ai.ai_model_split.bucket_splitter import BucketSplitPolicy

from lx_dtypes.models.base_models.base_model import AppBaseModel


# -----------------------------------------------------------------------------
# Strong “choice” types (Pydantic validation + IDE autocomplete)
# -----------------------------------------------------------------------------
BackboneName = Literal[
    "gastro_rn50",
    "resnet50_imagenet",
    "resnet50_random",
    "efficientnet_b0_imagenet",
]

DeviceName = Literal["auto", "cpu", "cuda"]


def _now_utc() -> AwareDatetime:
    """Central helper for timezone-aware UTC timestamps."""
    return datetime.now(timezone.utc)


def _find_repo_root_from_lx_ai(path: Path) -> Path:
    """
    Walk upwards from a file path until a directory named 'lx_ai' is found,
    then return its parent directory (repo root).

    Example:
        /home/admin/dev/lx-ai/lx_ai/ai_model_config/config.yaml
                                     ↑ lx_ai
        returns:
        /home/admin/dev/lx-ai
    """
    path = path.resolve()

    for parent in [path] + list(path.parents): #Python wants both operands to be lists when using +
        if parent.name == "lx_ai":
            return parent.parent.resolve()

    raise RuntimeError(
        f"Could not infer base_dir: no 'lx_ai' directory found above {path}"
    )


# -----------------------------------------------------------------------------
# Optional: typed dict export (“ddict” style) – matches lx_dtypes patterns
# -----------------------------------------------------------------------------
class TrainingConfigDataDict(TypedDict):
    dataset_uuid: str
    labelset_version_to_train: int
    treat_unlabeled_as_negative: bool

    base_dir: str
    training_root: str
    checkpoints_dir: str
    runs_dir: str
    create_dirs: bool

    backbone_name: str
    backbone_checkpoint: Optional[str]
    freeze_backbone: bool

    num_epochs: int
    batch_size: int
    val_split: float
    test_split: float

    lr_head: float
    lr_backbone: float
    gamma_focal: float
    alpha_focal: float

    use_scheduler: bool
    warmup_epochs: int
    min_lr: float

    device: str
    random_seed: int

    updated_at: str


class TrainingConfig(AppBaseModel):
    """
    Training configuration for lx-ai.

    Main goals:
    - Replaces the old Django-settings + dataclass approach
    - Loads from YAML using AppBaseModel.from_yaml_file()
    - Strict schema validation (extra fields forbidden)
    - Deterministic project-path resolution
    - Safe dumping for run metadata (ddict / json)
    """

    # -------------------------------------------------------------------------
    # Model config: keep AppBaseModel behavior, ensure strictness for this config
    # -------------------------------------------------------------------------
    model_config = AppBaseModel.model_config | ConfigDict(extra="forbid")

    # -------------------------------------------------------------------------
    # Constants (ClassVar means: not a field, not part of model_dump)
    # -------------------------------------------------------------------------
    DEFAULT_LABELSET_VERSION: ClassVar[int] = 2

    # -------------------------------------------------------------------------
    # Dataset selection / semantics
    # -------------------------------------------------------------------------
    dataset_uuid: str = Field(
        ...,
        description="UUID of the AIDataSet definition to train on (lx_dtypes model).",
    )


    # -------------------------------------------------------------------------
    # Data source selection
    # -------------------------------------------------------------------------
    data_source: Literal["jsonl", "postgres"] = Field(
        default="jsonl",
        description="Where training data comes from.",
    )
    
    # Used ONLY when data_source == 'postgres'
    '''dataset_id: int | None = Field(
        default=None,
        description="AIDataSet.id in PostgreSQL (required for postgres mode).",
    )'''

    dataset_ids: List[int] | None = Field(
        default=None,
        description="List of AIDataSet.id values in PostgreSQL (required for postgres mode).",
   )

    
    # Used ONLY when data_source == 'jsonl'
    jsonl_path: Path | None = Field(
        default=None,
    description="Path to legacy JSONL file (used in jsonl mode).",
    )

    labelset_id: int | None = None
    labelset_version_to_train: int = Field(
        default=DEFAULT_LABELSET_VERSION,
        ge=1,
        description="Train only on labels that belong to LabelSet.version == this.",
    )

    treat_unlabeled_as_negative: bool = Field(
        default=True,
        description=(
            "If True: missing annotations become negative (0) and included in loss. "
            "If False: missing remain unknown and masked out."
        ),
    )

    # -------------------------------------------------------------------------
    # Paths (no Django BASE_DIR; we compute from repo structure or source_file)
    # -------------------------------------------------------------------------
    base_dir: Path | None = Field(
        default=None,
        description=(
            "Repo root used to build training_root/checkpoints_dir/runs_dir. "
            "If None, inferred from config location (source_file)."
        ),
    )

    training_root: Path | None = Field(
        default=None,
        description="Root folder for training artifacts. Defaults to base_dir/data/model_training.",
    )

    checkpoints_dir: Path | None = Field(
        default=None,
        description="Defaults to training_root/checkpoints.",
    )

    runs_dir: Path | None = Field(
        default=None,
        description="Defaults to training_root/runs.",
    )

    create_dirs: bool = Field(
        default=True, 
        description="If True, create training_root/checkpoints_dir/runs_dir on validation.",
    )

    # -------------------------------------------------------------------------
    # Model selection
    # -------------------------------------------------------------------------
    backbone_name: BackboneName = Field(
        default="gastro_rn50",
        description="Which CNN backbone factory variant to train.",
    )

    backbone_checkpoint: Path | None = Field(
        default=None,
        description="Optional path to pretrained checkpoint (.pth).",
    )


    freeze_backbone: bool = Field(
        default=True,
        description="If True, backbone is frozen (head-only training).",
    )

    # -------------------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------------------
    num_epochs: int = Field(default=35, ge=1)
    batch_size: int = Field(default=32, ge=1)

    #val_split: float = Field(default=0.2, ge=0.0, le=1.0)
    #test_split: float = Field(default=0.1, ge=0.0, le=1.0)

    lr_head: float = Field(default=1e-3, gt=0)
    lr_backbone: float = Field(default=1e-4, gt=0)

    gamma_focal: float = Field(default=2.0, ge=0.0)
    alpha_focal: float = Field(default=0.25, ge=0.0, le=1.0)

    use_scheduler: bool = Field(default=True)
    warmup_epochs: int = Field(default=3, ge=0)
    min_lr: float = Field(default=1e-6, gt=0)

    device: DeviceName = Field(default="auto")
    random_seed: int = Field(default=42, ge=0)

    bucket_policy: BucketSplitPolicy = Field(...)

    # -------------------------------------------------------------------------
    # Stable bucket split policy (hash-based, immutable roles)
    # -------------------------------------------------------------------------
    save_bucket_snapshot: bool = Field(
    default=False,
    description="If True, save bucket snapshot for this run."
)
    # -------------------------------------------------------------------------
    # Meta field: useful for run metadata dumping (NOT created_at from base)
    # -------------------------------------------------------------------------
    updated_at: AwareDatetime = Field(default_factory=_now_utc)

    save_bucket_snapshot: bool = Field(
    default=False,
    description="If True, save bucket snapshot for this run."
)

    # -------------------------------------------------------------------------
    # Serializer: ISO string for JSON metadata
    # -------------------------------------------------------------------------
    @field_serializer("updated_at")
    def _serialize_updated_at(self, v: AwareDatetime) -> str:
        return v.isoformat()

    # -------------------------------------------------------------------------
    # ddict interface (matches lx_dtypes pattern)
    # -------------------------------------------------------------------------
    @property
    def ddict(self) -> type[TrainingConfigDataDict]:
        return TrainingConfigDataDict

    def to_ddict(self) -> TrainingConfigDataDict:
        # AppBaseModel.model_dump() already excludes source_file + created_at
        # but we also want to ensure Path fields are rendered as strings.
        data = self.model_dump()

        # In case json encoders don’t stringify Paths in your environment,
        # enforce conversion here (extra safe + stable meta dumps).
        for k in ("base_dir", "training_root", "checkpoints_dir", "runs_dir", "backbone_checkpoint"):
            if k in data and data[k] is not None:
                data[k] = str(data[k])


        return cast(TrainingConfigDataDict, data)

    # -------------------------------------------------------------------------
    # Validators: coerce strings → Path (YAML will usually supply strings)
    # -------------------------------------------------------------------------
    @field_validator("base_dir", "training_root", "checkpoints_dir", "runs_dir", mode="before")
    @classmethod
    def _coerce_to_path(cls, v: object) -> Path | None:
        if v is None or v == "":
            return None
        if isinstance(v, Path):
            return v.expanduser()
        if isinstance(v, str):
            return Path(v).expanduser()
        raise TypeError("Expected Path | str | None")
    
    
    @classmethod
    def _coerce_checkpoint_to_path(cls, v: object) -> Path | None:
        if v is None or v == "":
            return None
        if isinstance(v, Path):
            return v.expanduser()
        if isinstance(v, str):
            return Path(v).expanduser()
        raise TypeError("backbone_checkpoint must be a Path, str, or None")
 
    # -------------------------------------------------------------------------
    # Cross-field invariants & computed defaults (the “brain”)
    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def _fill_and_validate_paths(self) -> "TrainingConfig":
        """
        Fill base_dir/training_root/checkpoints_dir/runs_dir consistently.

        Priority:
          1) explicit training_root/checkpoints_dir/runs_dir if provided
          2) else derive from base_dir
          3) else infer base_dir from source_file (AppBaseModel)
        """
        # 1) Infer base_dir if missing
        if self.base_dir is None:
            if self.source_file is not None:
                # Best-case: config came from YAML → infer repo root from it
                self.base_dir = _find_repo_root_from_lx_ai(self.source_file)
            else:
                # Fallback: run from cwd
                self.base_dir = Path.cwd()

        self.base_dir = self.base_dir.resolve()

        # 2) training_root default
        if self.training_root is None:
            self.training_root = self.base_dir / "data" / "model_training"
            # self.training_root = self.training_root.resolve()

        # 3) child dirs defaults
        if self.checkpoints_dir is None:
            self.checkpoints_dir = self.training_root / "checkpoints"
        if self.runs_dir is None:
            self.runs_dir = self.training_root / "runs"

        self.checkpoints_dir = self.checkpoints_dir.resolve()
        self.runs_dir = self.runs_dir.resolve()

        # 4) Create directories if requested
        if self.create_dirs:
            for d in (self.training_root, self.checkpoints_dir, self.runs_dir):
                d.mkdir(parents=True, exist_ok=True)

        # 5) Validate split ratios (must leave room for training set)
        #if self.val_split + self.test_split >= 1.0:
        #    raise ValueError("val_split + test_split must be < 1.0")

        # 6) Validate checkpoint if provided
        if self.backbone_checkpoint is not None:
            ckpt = self.backbone_checkpoint.expanduser().resolve()
            if not ckpt.is_file():
                raise ValueError(f"backbone_checkpoint does not exist or is not a file: {ckpt}")
            self.backbone_checkpoint = ckpt

        # Update updated_at whenever config is validated (nice for traceability)
        self.updated_at = _now_utc()
        return self
    
    @model_validator(mode="after")
    def _validate_data_source(self) -> "TrainingConfig":
        if self.data_source == "postgres":
            if not self.dataset_ids:
                raise ValueError("dataset_ids must be provided when data_source='postgres'")
        if self.data_source == "jsonl":
            if self.jsonl_path is None:
                raise ValueError("jsonl_path must be set when data_source='jsonl'")
        return self

    
    @model_validator(mode="after")
    def _validate_labelset(self):
        if self.data_source == "postgres" and self.labelset_id is None:
            raise ValueError("labelset_id must be provided for postgres data source")
        return self
    