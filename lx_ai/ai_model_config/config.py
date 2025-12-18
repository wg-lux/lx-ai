from __future__ import annotations
from pydantic import (BaseModel, Field, validator )
from typing import Optional, Literal
from lx_dtypes.models.base_models.base_model import AppBaseModel


from pathlib import Path

class TrainingConfig(AppBaseModel):
    """
    Configuration for model training with dynamic parameters and configurations.
    This class inherits from AppBaseModel to leverage its UUID and serialization features.
    
    Fields:
    - model_name: Defines the name of the model to train (e.g., GastroNet, ResNet).
    - dataset_id: Dataset identifier for the AI dataset used for training.
    - labelset_version_to_train: Version of the LabelSet to use for training.
    - device: Device used for training (auto, cpu, cuda).
    - lr_head, lr_backbone: Learning rates for the head and backbone of the model.
    - num_epochs: Number of epochs for training.
    """

    # --- General Config ---
    model_name: Literal["gastro_rn50", "resnet50_imagenet", "efficientnet_b0_imagenet", "resnet50_random"] = "gastro_rn50"
    dataset_id: int  # The dataset to use for training
    labelset_version_to_train: int = 2  # Default version of labelset
    treat_unlabeled_as_negative: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # --- Hyperparameters ---
    num_epochs: int = 5
    batch_size: int = 32
    lr_head: float = 1e-3  # Learning rate for the head layer
    lr_backbone: float = 1e-4  # Learning rate for the backbone
    min_lr: float = 1e-6  # Minimum learning rate
    warmup_epochs: int = 3  # Number of warm-up epochs
    use_scheduler: bool = True  # Whether to use learning rate scheduler

    # --- Model-related ---
    backbone_checkpoint: Optional[str] = None  # Path to the pre-trained backbone model checkpoint
    freeze_backbone: bool = True  # Whether to freeze the backbone layers during training

    class Config:
        # --- Configuration for Pydantic ---
        # Enabling arbitrary types, which is useful for custom types like Path and UUID
        arbitrary_types_allowed = True
        # Always strip whitespace in strings to ensure uniformity
        str_strip_whitespace = True
        # Enforce validation of all fields (even defaults) to avoid incorrect data
        validate_default = True

    @validator('backbone_checkpoint', pre=True, always=True)
    def validate_checkpoint(cls, v):
        # Ensuring the checkpoint path exists (if it's provided)
        if v and not Path(v).exists():
            raise ValueError(f"The checkpoint path {v} does not exist.")
        return v

    @classmethod
    def from_ddict(cls, data: dict) -> "TrainingConfig":
        """
        Convert a dictionary to a TrainingConfig instance. This method can be used for
        deserializing from an external source (e.g., JSON, YAML, etc.)
        """
        return cls.model_validate(data)

    def model_dump(self, *args, **kwargs):
        """
        Override the default model_dump to handle some fields differently.
        This is useful for serialization, especially when dealing with file paths or timestamps.
        """
        kwargs.setdefault("exclude", {"created_at", "source_file"})  # Exclude unnecessary fields
        return super().model_dump(*args, **kwargs)

# Example usage:
# config = TrainingConfig(dataset_id=1, model_name="gastro_rn50", num_epochs=5)
