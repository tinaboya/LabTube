"""LabTube — VideoMAE Training Script (2 h/frame bin, 48 h window).

Expects an NPZ built with bin_hours=2.0 and window_hours=48.0:
  - NPZ video shape  : (24, 8, 8)  — 24 frames × 2 h = 48 h window
  - pixel_values     : (24, 2, 8, 8) — ch0 = z-scored lab value, ch1 = obs mask

Architecture (identical to original modeB but window covers 48 h):
  VideoMAEConfig: image_size=8, patch_size=2, num_frames=24, tubelet_size=2
  → 12 temporal × 16 spatial = 192 tokens per sample

Mode B pipeline:
  Stage 1: SSL pretraining  (VideoMAEForPreTraining)
  Stage 2: Fine-tune classification head  (VideoMAEForVideoClassification)

Usage
-----
  python scripts/labtube_train.py                        # all defaults
  python scripts/labtube_train.py --label los_3          # different label
  python scripts/labtube_train.py --pretrain-epochs 50   # shorter run
  python scripts/labtube_train.py --ckpt-dir results/run1
  python scripts/labtube_train.py --skip-pretrain        # fine-tune only
"""

from __future__ import annotations

import argparse
import io
import json
import math
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    VideoMAEConfig,
    VideoMAEForPreTraining,
    VideoMAEForVideoClassification,
)

# ── Default paths ─────────────────────────────────────────────────────────────
NPZ_PATH         = "data/labtube_dataset_2h.npz"
CKPT_DIR_DEFAULT = "results/checkpoints_2h"

# ── Token layout ──────────────────────────────────────────────────────────────
# num_frames=24, tubelet_size=2, image_size=8, patch_size=2
#   temporal tokens : 24 // 2 = 12
#   spatial  tokens : (8 // 2)² = 16
#   total           : 192
#
# FIXED_N_MASK: observed tokens to mask per sample during pretraining.
# Same as original (mean ~21 observed tokens / 192, obs_mask_ratio=0.5 
# → ~11).
FIXED_N_MASK = 11


# ── Helpers ───────────────────────────────────────────────────────────────────

def n_tokens(cfg: dict):
    t = cfg["num_frames"] // cfg["tubelet_size"]
    s = (cfg["image_size"] // cfg["patch_size"]) ** 2
    return t, s, t * s


def make_base_config(cfg: dict) -> VideoMAEConfig:
    return VideoMAEConfig(
        image_size                   = cfg["image_size"],
        patch_size                   = cfg["patch_size"],
        num_channels                 = cfg["num_channels"],
        num_frames                   = cfg["num_frames"],
        tubelet_size                 = cfg["tubelet_size"],
        hidden_size                  = cfg["hidden_size"],
        num_hidden_layers            = cfg["num_hidden_layers"],
        num_attention_heads          = cfg["num_attention_heads"],
        intermediate_size            = cfg["intermediate_size"],
        hidden_dropout_prob          = 0.1,
        attention_probs_dropout_prob = 0.1,
        norm_pix_loss                = True,
    )


def make_scheduler(optimizer, total_steps: int, warmup_ratio: float, base_lr: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _pos_weight(labels, device) -> torch.Tensor:
    pos = sum(labels)
    neg = len(labels) - pos
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_cls(model, loader, device, loss_fn):
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    for batch in loader:
        pv     = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(pixel_values=pv).logits.squeeze(-1)
        total_loss += loss_fn(logits, labels).item() * len(labels)
        all_logits.extend(logits.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    try:
        auroc = roc_auc_score(all_labels, probs)
        auprc = average_precision_score(all_labels, probs)
    except ValueError:
        auroc = auprc = float("nan")
    return total_loss / len(loader.dataset), auroc, auprc


# ── Datasets ──────────────────────────────────────────────────────────────────

class LabTubeDataset(Dataset):
    """Classification dataset.

    Expects NPZ videos of shape (24, 8, 8) with bin_hours=2 (48 h window).

    Returns:
        pixel_values : FloatTensor (24, 2, 8, 8)
        labels       : FloatTensor scalar
    """

    def __init__(self, npz_data, hadm_ids: list[int], labels_df: pd.DataFrame,
                 label_col: str, key_map: dict[int, str]):
        self.data     = npz_data
        self.key_map  = key_map
        lut           = labels_df.set_index("hadm_id")[label_col].to_dict()
        self.hadm_ids = [h for h in hadm_ids if h in key_map and h in lut]
        self.labels   = [float(lut[h]) for h in self.hadm_ids]

    def __len__(self):
        return len(self.hadm_ids)

    def __getitem__(self, idx):
        raw      = self.data[self.key_map[self.hadm_ids[idx]]].astype(np.float32)  # (24,8,8)
        obs_mask = (~np.isnan(raw)).astype(np.float32)
        raw      = np.nan_to_num(raw, nan=0.0)
        # Stack along new channel axis: (24, 2, 8, 8) = (T, C, H, W)
        pixel_values = np.stack([raw, obs_mask], axis=1)
        return {
            "pixel_values": torch.from_numpy(pixel_values),
            "labels":       torch.tensor(self.labels[idx]),
        }


class LabTubePretrainDataset(Dataset):
    """SSL pretraining dataset — no labels needed.

    Masking strategy: mask FIXED_N_MASK observed tokens per sample.
    Only observed tokens are masked; masking unobserved (zero-padded) tokens
    provides no learning signal.
    """

    def __init__(self, npz_data, hadm_ids: list[int], key_map: dict[int, str], cfg: dict):
        self.data         = npz_data
        self.key_map      = key_map
        self.hadm_ids     = [h for h in hadm_ids if h in key_map]
        _, _, total       = n_tokens(cfg)
        self.total_tokens = total
        self.n_mask       = FIXED_N_MASK
        self.tubelet_size = cfg["tubelet_size"]
        self.num_frames   = cfg["num_frames"]
        self.image_size   = cfg["image_size"]
        self.patch_size   = cfg["patch_size"]

    def __len__(self):
        return len(self.hadm_ids)

    def _observed_token_flags(self, obs_mask_3d: np.ndarray) -> np.ndarray:
        """Coarsen (24, 8, 8) obs_mask to (num_tokens,) bool.

        A token is considered observed if any pixel within its spatiotemporal
        tube (tubelet_size × patch_size × patch_size) is observed.
        """
        T  = self.num_frames  // self.tubelet_size   # 12
        ph = self.image_size  // self.patch_size      # 4
        pw = self.image_size  // self.patch_size      # 4
        m  = obs_mask_3d.reshape(T, self.tubelet_size, ph, self.patch_size, pw, self.patch_size)
        return m.any(axis=(1, 3, 5)).reshape(-1)     # (192,) bool

    def __getitem__(self, idx):
        raw      = self.data[self.key_map[self.hadm_ids[idx]]].astype(np.float32)  # (24,8,8)
        obs_mask = (~np.isnan(raw)).astype(np.float32)
        raw      = np.nan_to_num(raw, nan=0.0)
        pixel_values = np.stack([raw, obs_mask], axis=1)  # (24, 2, 8, 8)

        obs_tokens   = self._observed_token_flags(obs_mask)  # (192,) bool
        observed_idx = np.where(obs_tokens)[0]

        bool_masked = np.zeros(self.total_tokens, dtype=bool)
        if len(observed_idx) >= self.n_mask:
            chosen = np.random.choice(observed_idx, size=self.n_mask, replace=False)
        else:
            # Fallback: too sparse — random mask over all tokens
            chosen = np.random.choice(self.total_tokens, size=self.n_mask, replace=False)
        bool_masked[chosen] = True

        return {
            "pixel_values":    torch.from_numpy(pixel_values),
            "bool_masked_pos": torch.from_numpy(bool_masked),
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LabTube VideoMAE training (2 h/frame, 48 h window)")
    p.add_argument("--npz",             default=NPZ_PATH,         help="Path to NPZ (bin_hours=2)")
    p.add_argument("--ckpt-dir",        default=CKPT_DIR_DEFAULT, help="Checkpoint output directory")
    p.add_argument("--label",           default="in_hospital_mortality",
                   choices=["in_hospital_mortality", "los_3", "los_7", "short_los"],
                   help="Prediction target column")
    p.add_argument("--pretrain-epochs", type=int,   default=100)
    p.add_argument("--finetune-epochs", type=int,   default=50)
    p.add_argument("--pretrain-lr",     type=float, default=3e-4)
    p.add_argument("--finetune-lr",     type=float, default=3e-4)
    p.add_argument("--pretrain-batch",  type=int,   default=2048)
    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--skip-pretrain",   action="store_true",
                   help="Load existing modeB_pretrained.pt and skip to fine-tuning")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR = Path(args.ckpt_dir)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────────
    CFG = dict(
        label_col           = args.label,
        image_size          = 8,
        patch_size          = 2,
        num_channels        = 2,
        num_frames          = 24,          # 24 frames × 2 h = 48 h window
        tubelet_size        = 2,
        hidden_size         = 192,
        num_hidden_layers   = 4,
        num_attention_heads = 4,
        intermediate_size   = 768,
        pretrain_epochs     = args.pretrain_epochs,
        pretrain_lr         = args.pretrain_lr,
        pretrain_batch_size = args.pretrain_batch,
        obs_mask_ratio      = 0.5,
        finetune_epochs     = args.finetune_epochs,
        finetune_lr         = args.finetune_lr,
        batch_size          = args.batch_size,
        weight_decay        = 0.05,
        warmup_ratio        = 0.1,
        grad_clip           = 1.0,
    )

    t, s, total = n_tokens(CFG)
    print(f"Video layout  : {CFG['num_frames']} frames × 2 h = 48 h window")
    print(f"Tokens/sample : {t} temporal × {s} spatial = {total}")
    print(f"FIXED_N_MASK  : {FIXED_N_MASK} / {total} = {FIXED_N_MASK/total*100:.1f}% masked")
    print(f"Device        : {DEVICE}")
    print(f"Checkpoint dir: {CKPT_DIR}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading {args.npz} …")
    _npz       = np.load(args.npz, allow_pickle=False)
    labels_df  = pd.read_csv(io.BytesIO(_npz["meta_labels_csv"].tobytes()))
    split      = json.loads(_npz["meta_split_json"].tobytes())
    video_keys = [k for k in _npz.files if k.startswith("vid_")]
    hadm_ids   = [int(k[4:]) for k in video_keys]

    # Pre-load all videos into a plain dict so DataLoader workers don't share
    # the NpzFile's file handle (which causes BadZipFile across processes).
    print(f"  Pre-loading {len(video_keys):,} videos into memory …")
    data    = {k: _npz[k] for k in video_keys}
    key_map = {hid: k for hid, k in zip(hadm_ids, video_keys)}
    _npz.close()
    print(f"  Videos: {len(video_keys):,}  |  Train/Val/Test: "
          f"{len(split['train']):,}/{len(split['val']):,}/{len(split['test']):,}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def _make_cls_ds(ids):
        return LabTubeDataset(data, ids, labels_df, CFG["label_col"], key_map)

    train_ds = _make_cls_ds(split["train"])
    val_ds   = _make_cls_ds(split["val"])
    test_ds  = _make_cls_ds(split["test"])

    _mk = lambda ds, shuf: DataLoader(
        ds, batch_size=CFG["batch_size"], shuffle=shuf,
        num_workers=args.num_workers, pin_memory=True,
    )
    train_loader = _mk(train_ds, True)
    val_loader   = _mk(val_ds, False)
    test_loader  = _mk(test_ds, False)

    pos_rate = sum(train_ds.labels) / len(train_ds.labels)
    print(f"  Train {len(train_ds):,}  Val {len(val_ds):,}  Test {len(test_ds):,}")
    print(f"  Positive rate (train): {pos_rate:.3f}  pos_weight ≈ {(1-pos_rate)/pos_rate:.1f}")

    # ── Stage B1: SSL pretraining ─────────────────────────────────────────────
    pretrain_ckpt = CKPT_DIR / "modeB_pretrained.pt"
    config_b      = make_base_config(CFG)

    if args.skip_pretrain and pretrain_ckpt.exists():
        print(f"\n[B1] Skipping pretraining — loading {pretrain_ckpt}")
        model_b_pretrain = VideoMAEForPreTraining(config_b).to(DEVICE)
        ck = torch.load(pretrain_ckpt, map_location=DEVICE)
        model_b_pretrain.load_state_dict(ck["state_dict"])
    else:
        pretrain_all_ids = split["train"] + split["val"]
        pretrain_ds = LabTubePretrainDataset(data, pretrain_all_ids, key_map, CFG)
        pretrain_loader = DataLoader(
            pretrain_ds, batch_size=CFG["pretrain_batch_size"],
            shuffle=True, num_workers=args.num_workers, pin_memory=True,
        )
        print(f"\n[B1] Pretraining on {len(pretrain_ds):,} admissions …")

        model_b_pretrain = VideoMAEForPreTraining(config_b).to(DEVICE)
        n_params = sum(p.numel() for p in model_b_pretrain.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params/1e6:.2f}M")

        optimizer_pre   = torch.optim.AdamW(
            model_b_pretrain.parameters(), lr=CFG["pretrain_lr"], weight_decay=CFG["weight_decay"]
        )
        total_steps_pre = CFG["pretrain_epochs"] * len(pretrain_loader)
        scheduler_pre   = make_scheduler(optimizer_pre, total_steps_pre, CFG["warmup_ratio"], CFG["pretrain_lr"])

        pretrain_history = []
        for epoch in tqdm(range(1, CFG["pretrain_epochs"] + 1)):
            model_b_pretrain.train()
            epoch_loss = 0.0
            for batch in pretrain_loader:
                pv  = batch["pixel_values"].to(DEVICE)
                bmp = batch["bool_masked_pos"].to(DEVICE)
                optimizer_pre.zero_grad()
                loss = model_b_pretrain(pixel_values=pv, bool_masked_pos=bmp).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_b_pretrain.parameters(), CFG["grad_clip"])
                optimizer_pre.step()
                scheduler_pre.step()
                epoch_loss += loss.item() * len(pv)
            epoch_loss /= len(pretrain_ds)
            pretrain_history.append({"epoch": epoch, "loss": epoch_loss})
            print(f"  [pretrain] Epoch {epoch:3d}  recon_loss={epoch_loss:.4f}")
            torch.save({"state_dict": model_b_pretrain.state_dict(), "epoch": epoch}, pretrain_ckpt)

        pd.DataFrame(pretrain_history).to_csv(CKPT_DIR / "modeB_pretrain_history.csv", index=False)
        print(f"[B1] Done. Checkpoint → {pretrain_ckpt}")

    # ── Stage B2: Transfer weights ────────────────────────────────────────────
    print("\n[B2] Transferring encoder weights …")
    config_b_cls            = make_base_config(CFG)
    config_b_cls.num_labels = 1
    model_b_cls = VideoMAEForVideoClassification(config_b_cls).to(DEVICE)

    pretrain_sd = model_b_pretrain.state_dict()
    cls_sd      = model_b_cls.state_dict()
    transferred = {k: v for k, v in pretrain_sd.items() if k in cls_sd}
    cls_sd.update(transferred)
    model_b_cls.load_state_dict(cls_sd)
    print(f"  Transferred {len(transferred)}/{len(cls_sd)} parameter tensors")

    # ── Stage B3: Fine-tune classification ───────────────────────────────────
    print(f"\n[B3] Fine-tuning for '{CFG['label_col']}' …")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=_pos_weight(train_ds.labels, DEVICE))

    encoder_params = [p for n, p in model_b_cls.named_parameters() if "classifier" not in n]
    head_params    = [p for n, p in model_b_cls.named_parameters() if "classifier" in n]
    optimizer_ft   = torch.optim.AdamW([
        {"params": encoder_params, "lr": CFG["finetune_lr"] * 0.1},   # 10× lower: avoid forgetting
        {"params": head_params,    "lr": CFG["finetune_lr"]},
    ], weight_decay=CFG["weight_decay"])

    total_steps_ft = CFG["finetune_epochs"] * len(train_loader)
    scheduler_ft   = make_scheduler(optimizer_ft, total_steps_ft, CFG["warmup_ratio"], CFG["finetune_lr"])

    best_auroc, history = 0.0, []
    best_ckpt = CKPT_DIR / "modeB_best.pt"

    for epoch in tqdm(range(1, CFG["finetune_epochs"] + 1)):
        model_b_cls.train()
        train_loss = 0.0
        for batch in train_loader:
            pv, labels = batch["pixel_values"].to(DEVICE), batch["labels"].to(DEVICE)
            optimizer_ft.zero_grad()
            loss = loss_fn(model_b_cls(pixel_values=pv).logits.squeeze(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_b_cls.parameters(), CFG["grad_clip"])
            optimizer_ft.step()
            scheduler_ft.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_ds)

        val_loss, val_auroc, val_auprc = evaluate_cls(model_b_cls, val_loader, DEVICE, loss_fn)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            val_auroc=val_auroc, val_auprc=val_auprc))
        print(f"  [finetune] Epoch {epoch:3d}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"auroc={val_auroc:.4f}  auprc={val_auprc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({"epoch": epoch, "state_dict": model_b_cls.state_dict(),
                        "val_auroc": val_auroc, "cfg": CFG}, best_ckpt)
            print(f"            → saved best (auroc={val_auroc:.4f})")

    pd.DataFrame(history).to_csv(CKPT_DIR / "modeB_finetune_history.csv", index=False)
    print(f"\n[B3] Done. Best val AUROC: {best_auroc:.4f}  Checkpoint → {best_ckpt}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n[Test] Evaluating on held-out test set …")
    ck = torch.load(best_ckpt, map_location=DEVICE)
    model_b_cls.load_state_dict(ck["state_dict"])
    test_loss, test_auroc, test_auprc = evaluate_cls(model_b_cls, test_loader, DEVICE, loss_fn)
    print(f"  Test loss={test_loss:.4f}  AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}")

    result = {"test_loss": test_loss, "test_auroc": test_auroc, "test_auprc": test_auprc}
    with open(CKPT_DIR / "test_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved → {CKPT_DIR / 'test_results.json'}")


if __name__ == "__main__":
    main()
