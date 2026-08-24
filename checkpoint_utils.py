import os

import torch


REQUIRED_TRAINING_CHECKPOINT_KEYS = {
    "model_state_dict",
    "optimizer_state_dict",
    "scaler_state_dict",
    "net_config",
    "iteration",
    "train_step",
    "global_step",
}


def load_training_checkpoint(path, map_location="cpu"):
    """Load and validate a checkpoint produced by the current trainer."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"training checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("training checkpoint must be a dictionary")

    missing = sorted(REQUIRED_TRAINING_CHECKPOINT_KEYS.difference(checkpoint))
    if missing:
        raise KeyError(f"training checkpoint is missing required keys: {missing}")

    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("model_state_dict must be a non-empty dictionary")

    compiled_keys = [key for key in state_dict if key.startswith("_orig_mod.")]
    if compiled_keys:
        raise ValueError(
            "checkpoint model_state_dict must use canonical uncompiled keys; "
            f"found compiled key {compiled_keys[0]!r}"
        )

    return checkpoint


def restore_model_state_strict(model, checkpoint):
    """Restore exact model parameters before torch.compile wraps the model."""
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def canonical_model_state_dict(model, *, to_cpu=False):
    """Return a state dict whose keys are independent of torch.compile wrappers."""
    canonical = {}
    for key, value in model.state_dict().items():
        clean_key = key.removeprefix("_orig_mod.")
        if clean_key in canonical:
            raise ValueError(f"duplicate canonical model key: {clean_key!r}")
        canonical[clean_key] = value.cpu() if to_cpu else value
    return canonical
