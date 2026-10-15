"""openpi integration (openpi env only): transforms, data config, TrainConfigs.

The only patched seam is `openpi.training.data_loader.create_torch_dataset`:
repo_ids starting with "roboeval/" build a RoboEvalParquetDataset using the
factory stashed in _REPO_FACTORIES (DataConfig itself is frozen/generic).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence

import numpy as np

DATASET_ROOT = Path("/workspace/roboeval/data/roboeval_ee_delta_20hz_scaled")
REGISTRY_PATH = (Path(__file__).resolve().parents[3]
                 / "data_manifests" / "demo_registry.json")
ASSETS_BASE = "/workspace/roboeval/models/openpi_assets"
CHECKPOINT_BASE = "/workspace/roboeval/models/checkpoints"
PYTORCH_WEIGHT_PATH = "/workspace/roboeval/models/pi05_base_pytorch"

PR8_VARIANTS = (
    "LiftPotPositionAndOrientation", "LiftTrayPositionAndOrientation",
    "StackTwoBlocksPositionAndOrientation", "StackSingleBookShelfPositionAndOrientation",
    "CubeHandoverPositionAndOrientation", "PackBoxPositionAndOrientation",
    "PickSingleBookFromTablePositionAndOrientation", "RotateValvePositionAndOrientation",
)

_REPO_FACTORIES: dict[str, "RoboEvalDataConfig"] = {}

# Correlated-noise assets per config name (Larchenko et al. §5, beta=0.5).
CORR_NOISE_ASSETS = {
    "pi05_roboeval_pr8_corr": (ASSETS_BASE + "/pi05_roboeval_pr8/roboeval/"
                             "pr8_qpos18_h16/action_corr_beta0.5.npz"),
}


def maybe_install_corr_noise(config_name: str) -> "object | None":
    """Load + patch correlated noise for configs present in CORR_NOISE_ASSETS.

    Returns the CorrelatedNoise or None. Raises FileNotFoundError if the npz
    is missing — no silent fallback.
    """
    path = CORR_NOISE_ASSETS.get(config_name)
    if path is None:
        return None
    if not Path(path).exists():
        raise FileNotFoundError(f"corr noise asset missing: {path}")
    from roboeval_entry.training.corr_noise import (CorrelatedNoise,
                                                    patch_pi0_sample_noise)
    cn = CorrelatedNoise.load(path)
    patch_pi0_sample_noise(cn)
    print(f"[corr_noise] patched sample_noise: beta={cn.beta}, "
          f"corr={cn.corr.shape}, asset={path}", flush=True)
    return cn


def _require_openpi():
    import openpi.models.model as _model  # noqa: F401
    import openpi.transforms as _transforms

    return _model, _transforms


class RoboEvalInputs:
    """Dataset raw layout -> model input dict.

    Plain module-level callable (DataTransformFn requires openpi; duck-typing
    suffices for transforms.Group and keeps it picklable for workers).
    """

    def __init__(self, model_type=None):
        self.model_type = model_type

    def __call__(self, data: dict) -> dict:
        image = {
            "base_0_rgb": np.asarray(data["observation/head_image"]),
            "left_wrist_0_rgb": np.asarray(data["observation/left_wrist_image"]),
            "right_wrist_0_rgb": np.asarray(data["observation/right_wrist_image"]),
        }
        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": image,
            "image_mask": {k: np.True_ for k in image},
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


class RoboEvalOutputs:
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :14])}


def _repack_group():
    from openpi import transforms as _transforms

    return _transforms.Group(inputs=[_transforms.RepackTransform({
        "observation/head_image": "observation/head_image",
        "observation/left_wrist_image": "observation/left_wrist_image",
        "observation/right_wrist_image": "observation/right_wrist_image",
        "observation/state": "observation/state",
        "actions": "actions",
        "actions_is_pad": "actions_is_pad",
        "prompt": "prompt",
    })])


def _data_config_factory_cls():
    import openpi.training.config as _config
    import openpi.transforms as _transforms

    @dataclasses.dataclass(frozen=True)
    class RoboEvalDataConfig(_config.DataConfigFactory):
        repo_id: str = "roboeval/pr8_qpos18_h16"
        dataset_root: str = str(DATASET_ROOT)
        registry_path: str = str(REGISTRY_PATH)
        variants: tuple[str, ...] = PR8_VARIANTS
        split: str = "train"
        episode_limit: int | None = None
        action_horizon_override: int | None = None

        def create(self, assets_dirs, model_config):
            model_transforms = _config.ModelTransformFactory()(model_config)
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=_repack_group(),
                data_transforms=_transforms.Group(
                    inputs=[RoboEvalInputs(model_config.model_type)],
                    outputs=[RoboEvalOutputs()],
                ),
                model_transforms=model_transforms,
                prompt_from_task=False,
                action_sequence_keys=("actions",),
            )

    return RoboEvalDataConfig


RoboEvalDataConfig = None  # populated by install()


def _create_dataset_for_repo(data_config, action_horizon, model_config):
    from roboeval_entry.training.parquet_dataset import RoboEvalParquetDataset

    factory = _REPO_FACTORIES[data_config.repo_id]
    return RoboEvalParquetDataset(
        dataset_root=factory.dataset_root,
        variants=factory.variants,
        split=factory.split,
        registry_path=factory.registry_path,
        action_horizon=action_horizon,
        episode_limit=factory.episode_limit,
    )


def install() -> None:
    """Patch the dataset seam and register our TrainConfigs. Idempotent."""
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.training.optimizer as _optimizer

    if getattr(_data_loader.create_torch_dataset, "_roboeval_patched", False):
        return

    RDC = _data_config_factory_cls()
    globals()["RoboEvalDataConfig"] = RDC

    overfit_factory = RDC(repo_id="roboeval/overfit_liftpot4",
                          variants=("LiftPotPositionAndOrientation",),
                          episode_limit=4, split="train")
    pr8_factory = RDC(repo_id="roboeval/pr8_qpos18_h16", variants=PR8_VARIANTS,
                      split="train")
    _REPO_FACTORIES["roboeval/overfit_liftpot4"] = overfit_factory
    _REPO_FACTORIES["roboeval/pr8_qpos18_h16"] = pr8_factory

    orig_create = _data_loader.create_torch_dataset

    def patched(data_config, action_horizon, model_config):
        if data_config.repo_id and data_config.repo_id.startswith("roboeval/"):
            return _create_dataset_for_repo(data_config, action_horizon, model_config)
        return orig_create(data_config, action_horizon, model_config)

    patched._roboeval_patched = True  # type: ignore[attr-defined]
    _data_loader.create_torch_dataset = patched

    model = pi0_config.Pi0Config(pi05=True, action_horizon=16,
                                 discrete_state_input=True)
    _config._CONFIGS_DICT["pi05_roboeval_overfit"] = _config.TrainConfig(
        name="pi05_roboeval_overfit",
        model=model,
        data=overfit_factory,
        pytorch_weight_path=PYTORCH_WEIGHT_PATH,
        assets_base_dir=ASSETS_BASE,
        checkpoint_base_dir=CHECKPOINT_BASE,
        batch_size=16,
        num_train_steps=400,
        log_interval=10,
        save_interval=200,
        num_workers=8,
        seed=0,
        ema_decay=None,
        wandb_enabled=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=20, peak_lr=5e-5, decay_steps=400, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    )
    _config._CONFIGS_DICT["pi05_roboeval_pr8"] = _config.TrainConfig(
        name="pi05_roboeval_pr8",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=16,
                                   discrete_state_input=True),
        data=pr8_factory,
        pytorch_weight_path=PYTORCH_WEIGHT_PATH,
        assets_base_dir=ASSETS_BASE,
        checkpoint_base_dir=CHECKPOINT_BASE,
        batch_size=16,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        num_workers=16,
        seed=0,
        ema_decay=None,
        wandb_enabled=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=20_000,
            decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    )
    # Correlated-noise twin of pi05_roboeval_pr8: identical training config;
    # assets_dir points at the pr8 assets so the same norm_stats.json is
    # reused (asset_id falls back to repo_id "roboeval/pr8_qpos18_h16").
    corr_factory = dataclasses.replace(
        pr8_factory,
        assets=_config.AssetsConfig(
            assets_dir=f"{ASSETS_BASE}/pi05_roboeval_pr8"))
    _config._CONFIGS_DICT["pi05_roboeval_pr8_corr"] = _config.TrainConfig(
        name="pi05_roboeval_pr8_corr",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=16,
                                   discrete_state_input=True),
        data=corr_factory,
        pytorch_weight_path=PYTORCH_WEIGHT_PATH,
        assets_base_dir=ASSETS_BASE,
        checkpoint_base_dir=CHECKPOINT_BASE,
        batch_size=16,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        num_workers=16,
        seed=0,
        ema_decay=None,
        wandb_enabled=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=20_000,
            decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    )
