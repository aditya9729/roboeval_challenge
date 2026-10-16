"""Validated pi0.5 training loop (openpi env) + plateau annealing + HPO plumbing.

Validation tiers (settled):
- fit: `train` demos (registry split).
- val (HPO objective, plateau/early-stop signal): the 76 `dev` demos
  (registry split 'dev') — open-loop, no simulator.
- dev seeds (configs/dev_eval_seeds.json, pool 200000-200099):
  official-evaluator checkpoint selection (E010 style).
- holdout seeds (configs/holdout_seeds.json, 300000-300099): local test;
  read once per major decision, never for selection.
- official 1000-1049: never.

Checkpoint format is identical to vendor scripts/train_pytorch.py so
`create_trained_policy` and our serving load it unchanged. openpi PyTorch
has no EMA — the weights validated are the weights saved.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class ValidationConfig:
    interval: int = 250
    n_loss: int = 256
    n_openloop: int = 64
    patience: int = 3
    anneal_factor: float = 0.5
    min_lr_scale: float = 0.1
    max_anneals: int = 3
    rel_tol: float = 0.01
    seed: int = 0
    enabled: bool = True


class PlateauAnnealer:
    """Anneal lr_scale when the objective plateaus; signal early stop."""

    def __init__(self, patience: int, anneal_factor: float,
                 min_lr_scale: float, max_anneals: int, rel_tol: float):
        self.patience = patience
        self.anneal_factor = anneal_factor
        self.min_lr_scale = min_lr_scale
        self.max_anneals = max_anneals
        self.rel_tol = rel_tol
        self.best = math.inf
        self.bad = 0
        self.anneals = 0
        self.lr_scale = 1.0

    def update(self, objective: float) -> str:
        """Return 'improved' | 'no_improve' | 'anneal' | 'stop'."""
        if objective < self.best * (1 - self.rel_tol):
            self.best = objective
            self.bad = 0
            return "improved"
        self.bad += 1
        if self.bad < self.patience:
            return "no_improve"
        if self.anneals >= self.max_anneals:
            return "stop"
        self.anneals += 1
        self.bad = 0
        self.lr_scale = max(self.min_lr_scale,
                            self.lr_scale * self.anneal_factor)
        return "anneal"

    def to_dict(self) -> dict:
        return {"best": None if self.best == math.inf else self.best,
                "bad": self.bad, "anneals": self.anneals,
                "lr_scale": self.lr_scale}

    @classmethod
    def from_dict(cls, d: dict, **kwargs) -> "PlateauAnnealer":
        a = cls(**kwargs)
        a.best = d.get("best") or math.inf
        a.bad = d.get("bad", 0)
        a.anneals = d.get("anneals", 0)
        a.lr_scale = d.get("lr_scale", 1.0)
        return a


def openloop_objective(preds: np.ndarray, targets: np.ndarray,
                       mean_chunk: np.ndarray) -> tuple[float, float]:
    """Same objective as scripts/openloop_check.py: cont-mse ratio on arm
    dims 0-11 vs the per-dim mean-chunk baseline, plus gripper agreement."""
    cont = list(range(12))
    mse_pred = ((preds - targets) ** 2).mean(axis=(0, 1))
    base = np.broadcast_to(mean_chunk, targets.shape)
    mse_base = ((base - targets) ** 2).mean(axis=(0, 1))
    grip = ((preds[..., 12:14] > 0.5) == (targets[..., 12:14] > 0.5)).mean()
    return float(mse_pred[cont].mean() / mse_base[cont].mean()), float(grip)


def validation_noise(seed: int, n: int, horizon: int, dim: int):
    """Deterministic per-validation noise from a dedicated generator —
    never touches the global torch RNG."""
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, 1, horizon, dim), generator=g,
                       dtype=torch.float32), g


def save_step_checkpoint(model, optimizer, dest: Path, config, data_config,
                         global_step: int) -> None:
    """Same on-disk format as vendor save_checkpoint (tmp dir + rename)."""
    import torch
    import safetensors.torch
    import openpi.shared.normalize as _normalize

    tmp = dest.parent / f"tmp_{dest.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model_to_save = (model.module
                     if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                     else model)
    safetensors.torch.save_model(model_to_save, tmp / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save({"global_step": global_step,
                "config": dataclasses.asdict(config),
                "timestamp": time.time()}, tmp / "metadata.pt")
    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp / "assets" / data_config.asset_id,
                        data_config.norm_stats)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)


class Validator:
    """Dev-split validation: flow loss with fixed noise/time + deterministic
    open-loop objective. Never touches the global torch RNG."""

    def __init__(self, config, model, data_config, cfg: ValidationConfig,
                 device):
        import dataclasses as _dc
        import torch
        import openpi.training.data_loader as dl
        import openpi.transforms as _transforms
        from roboeval_entry.training.parquet_dataset import (
            RoboEvalParquetDataset)

        self.cfg = cfg
        self.device = device
        dev_factory = _dc.replace(config.data, split="dev")
        dev_dc = dev_factory.create(config.assets_dirs, config.model)
        raw_dev = dl.create_torch_dataset(dev_dc, config.model.action_horizon,
                                          config.model)
        self.raw_dev = raw_dev
        self.tds = dl.transform_dataset(raw_dev, dev_dc)
        n = len(raw_dev)
        self.loss_idxs = np.linspace(
            0, n - 1, min(cfg.n_loss, n), dtype=int)
        self.ol_idxs = np.linspace(
            0, n - 1, min(cfg.n_openloop, n), dtype=int)
        if min(cfg.n_loss, cfg.n_openloop) > n:
            logging.warning("dev dataset (%d) smaller than val n; clamped", n)
        # mean action chunk baseline over dev episodes (raw space)
        all_actions = np.concatenate(
            [ep["actions"] for ep in raw_dev._episodes])
        self.mean_chunk = all_actions.mean(axis=0).astype(np.float64)

        # input/output transforms for per-sample open-loop inference
        self._in_t = _transforms.compose([
            *dev_dc.repack_transforms.inputs,
            *dev_dc.data_transforms.inputs,
            _transforms.Normalize(dev_dc.norm_stats,
                                  use_quantiles=dev_dc.use_quantile_norm),
            *dev_dc.model_transforms.inputs,
        ])
        self._out_t = _transforms.compose([
            *dev_dc.model_transforms.outputs,
            _transforms.Unnormalize(dev_dc.norm_stats,
                                    use_quantiles=dev_dc.use_quantile_norm),
            *dev_dc.data_transforms.outputs,
        ])

        # fixed noise/time tensors (generated once; dedicated generators)
        H, D = config.model.action_horizon, config.model.action_dim
        n4, g = validation_noise(cfg.seed, len(self.loss_idxs), H, D)
        self._loss_noise = n4[:, 0]
        tb = np.random.default_rng(cfg.seed).beta(
            1.5, 1.0, len(self.loss_idxs)) * 0.999 + 0.001
        self._loss_time = torch.from_numpy(tb.astype(np.float32))
        self._ol_noise, _ = validation_noise(cfg.seed, len(self.ol_idxs),
                                             H, D)
        self._model = model

    def run(self, step: int, train_loss_avg: float, lr: float,
            lr_scale: float) -> dict:
        import jax
        import torch
        import openpi.models.model as _model

        model, cfg, device = self._model, self.cfg, self.device
        model.eval()
        t0 = time.time()

        # (a) val flow loss with fixed noise/time
        losses = []
        bs = 32
        with torch.no_grad():
            for s in range(0, len(self.loss_idxs), bs):
                idxs = self.loss_idxs[s:s + bs]
                items = [self.tds[int(i)] for i in idxs]
                batch = jax.tree.map(
                    lambda *xs: np.stack([np.asarray(x) for x in xs]),
                    *items)
                batch = jax.tree.map(
                    lambda x: torch.as_tensor(x).to(device), batch)
                obs = _model.Observation.from_dict(batch)
                actions = batch["actions"].to(torch.float32)
                b = actions.shape[0]
                sl = slice(s, s + b)
                per = model(obs, actions,
                            noise=self._loss_noise[sl].to(device),
                            time=self._loss_time[sl].to(device))
                losses.append(float(per.mean()))
        val_loss = float(np.mean(losses))

        # (b) deterministic open-loop objective on raw dev samples
        preds, targets = [], []
        with torch.no_grad():
            for i, idx in enumerate(self.ol_idxs):
                raw = self.raw_dev[int(idx)]
                ti = self._in_t(dict(raw))
                inputs = jax.tree.map(
                    lambda x: torch.from_numpy(np.array(x)).to(device)[None],
                    {k: ti[k] for k in
                     ("state", "image", "image_mask", "tokenized_prompt",
                      "tokenized_prompt_mask") if k in ti})
                obs = _model.Observation.from_dict(inputs)
                acts = model.sample_actions(device, obs,
                                            noise=self._ol_noise[i].to(device))
                out = self._out_t({"state": np.asarray(
                                       inputs["state"][0].detach().cpu()),
                                   "actions": np.asarray(
                                       acts[0].detach().cpu().numpy())})
                preds.append(np.asarray(out["actions"]))
                targets.append(np.asarray(raw["actions"]))
        cont_ratio, grip = openloop_objective(
            np.stack(preds), np.stack(targets), self.mean_chunk)
        model.train()
        return {"step": step, "val_loss": val_loss,
                "cont_mse_ratio": cont_ratio, "gripper_agreement": grip,
                "lr_scale": lr_scale, "lr": lr,
                "train_loss_avg": train_loss_avg,
                "wall_s": time.time() - t0}


def train_loop(config, val_cfg: ValidationConfig | None = None) -> None:
    """Single-process training loop (vendor train_pytorch.py shape)."""
    import jax
    import numpy as _np  # noqa
    import torch
    import tqdm
    import openpi.models.pi0_config
    import openpi.models_pytorch.pi0_pytorch
    import openpi.training.data_loader as _data

    val_cfg = val_cfg or ValidationConfig(enabled=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ckpt_dir = config.checkpoint_dir
    resuming = False
    if config.resume:
        if (ckpt_dir / "train_state.json").exists() or any(
                d.name.isdigit() for d in ckpt_dir.iterdir() if d.is_dir()):
            resuming = True
        else:
            raise FileNotFoundError(f"nothing to resume in {ckpt_dir}")
    elif config.overwrite and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if not resuming:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_loader = _data.create_data_loader(config, framework="pytorch",
                                           shuffle=True)
    loader = data_loader
    data_config = data_loader.data_config()

    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if config.pytorch_weight_path is not None:
        import safetensors.torch
        safetensors.torch.load_model(
            model, str(Path(config.pytorch_weight_path) / "model.safetensors"))
        logging.info("Loaded weights from %s", config.pytorch_weight_path)

    optim = torch.optim.AdamW(
        model.parameters(), lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps, weight_decay=config.optimizer.weight_decay)

    annealer = PlateauAnnealer(val_cfg.patience, val_cfg.anneal_factor,
                               val_cfg.min_lr_scale, val_cfg.max_anneals,
                               val_cfg.rel_tol)
    state_path = ckpt_dir / "train_state.json"
    global_step = 0
    if resuming:
        import safetensors.torch
        steps = [int(d.name) for d in ckpt_dir.iterdir()
                 if d.is_dir() and d.name.isdigit()]
        latest = max(steps)
        safetensors.torch.load_model(
            model, str(ckpt_dir / str(latest) / "model.safetensors"))
        optim.load_state_dict(torch.load(
            ckpt_dir / str(latest) / "optimizer.pt", map_location=device,
            weights_only=False))
        global_step = latest
        if state_path.exists():
            st = json.loads(state_path.read_text())
            annealer = PlateauAnnealer.from_dict(
                st["annealer"], patience=val_cfg.patience,
                anneal_factor=val_cfg.anneal_factor,
                min_lr_scale=val_cfg.min_lr_scale,
                max_anneals=val_cfg.max_anneals, rel_tol=val_cfg.rel_tol)

    warm, peak, dec, end = (config.lr_schedule.warmup_steps,
                            config.lr_schedule.peak_lr,
                            config.lr_schedule.decay_steps,
                            config.lr_schedule.decay_lr)

    def lr_schedule(step):
        if step < warm:
            init = peak / (warm + 1)
            return init + (peak - init) * step / warm
        prog = min(1.0, (step - warm) / max(1, dec - warm))
        return end + (peak - end) * 0.5 * (1 + np.cos(np.pi * prog))

    validator = None
    if val_cfg.enabled:
        validator = Validator(config, model, data_config, val_cfg, device)

    history_path = ckpt_dir / "val_history.jsonl"

    def persist_state():
        state_path.write_text(json.dumps(
            {"annealer": annealer.to_dict(), "step": global_step}))

    def maybe_validate() -> bool:
        if validator is None or not (
                global_step % val_cfg.interval == 0
                or global_step == config.num_train_steps):
            return True
        train_avg = float(np.mean(losses_since_val)) if losses_since_val \
            else float("nan")
        cur_lr = lr_schedule(global_step) * annealer.lr_scale
        res = validator.run(global_step, train_avg, cur_lr,
                            annealer.lr_scale)
        with open(history_path, "a") as f:
            f.write(json.dumps(res) + "\n")
        logging.info("val step=%d loss=%.4f cont_mse_ratio=%.3f grip=%.3f "
                     "lr_scale=%.3f", global_step, res["val_loss"],
                     res["cont_mse_ratio"], res["gripper_agreement"],
                     annealer.lr_scale)
        losses_since_val.clear()
        outcome = annealer.update(res["cont_mse_ratio"])
        if outcome == "improved":
            save_step_checkpoint(model, optim, ckpt_dir / "best", config,
                                 data_config, global_step)
            (ckpt_dir / "best" / "val.json").write_text(json.dumps(
                {"step": global_step, "objective": res["cont_mse_ratio"],
                 "val_loss": res["val_loss"]}))
        elif outcome == "anneal":
            logging.info("plateau: anneal #%d, lr_scale -> %.3f",
                         annealer.anneals, annealer.lr_scale)
        persist_state()
        return outcome != "stop"

    model.train()
    losses_since_val: list[float] = []
    infos: list[dict] = []
    pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step,
                     desc="Training")
    start_time = time.time()

    # validation at step 0 before any gradient updates
    if validator is not None and not resuming:
        if not maybe_validate():
            logging.info("early stop at step 0")
            return

    while global_step < config.num_train_steps:
        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)

            cur_lr = lr_schedule(global_step) * annealer.lr_scale
            for pg in optim.param_groups:
                pg["lr"] = cur_lr

            losses = model(observation, actions)
            loss = torch.stack(losses).mean() if isinstance(
                losses, (list, tuple)) else losses.mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.optimizer.clip_gradient_norm)
            optim.step()
            optim.zero_grad(set_to_none=True)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad = None

            losses_since_val.append(loss.item())
            infos.append({"loss": loss.item(), "lr": cur_lr,
                          "grad_norm": float(grad_norm)
                          if isinstance(grad_norm, torch.Tensor)
                          else grad_norm})
            if global_step % config.log_interval == 0:
                elapsed = time.time() - start_time
                avg = sum(i["loss"] for i in infos) / len(infos)
                logging.info("step=%d loss=%.4f lr=%.2e grad_norm=%.2f "
                             "time=%.1fs", global_step, avg,
                             cur_lr, infos[-1]["grad_norm"], elapsed)
                start_time = time.time()
                infos = []

            global_step += 1
            if (global_step % config.save_interval == 0
                    and global_step > 0) or \
                    global_step == config.num_train_steps - 1:
                save_step_checkpoint(model, optim,
                                     ckpt_dir / str(global_step), config,
                                     data_config, global_step)
            pbar.update(1)
            if not maybe_validate():
                logging.info("early stop at step %d (max anneals reached)",
                             global_step)
                pbar.close()
                return
    pbar.close()
