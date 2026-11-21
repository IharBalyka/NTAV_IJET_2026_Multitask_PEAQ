#!/usr/bin/env python
"""
peaq_predict.py — unified script for all architectures and sizes.

Launch:
  python peaq_predict.py                      # with UI
  python peaq_predict.py -i audio_dir -m models_dir -o results.csv

Result: CSV (one line for each audio, one column for each model-metrics
combination). If more than one model produces the same metric, the script
also adds MEAN[metric] columns.
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import subprocess
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import librosa

TARGET_SR = 44100
WINDOW_SEC = 7.0
OVERLAP = 0.5
N_FFT = 2048
HOP_LENGTH = 256

AUDIO_EXT = {".wav", ".mp3", ".ogg", ".opus", ".mp4", ".m4a", ".aac", ".flac"}

PEAQ13 = ["ODG", "DI"] + [f"MOV{i}" for i in range(1, 12)]

# PEAQ z-score stats (13 values, order = PEAQ13: ODG, DI, MOV1..MOV11).
DEFAULT_PEAQ_MEAN = np.array([
    -2.154414415359497, -0.4764358699321747, 858.1868286132812,
    711.8399658203125,  -5.66736364364624,   32.59767532348633,
    1.9103832244873047, 0.8361066579818726,  29.6385555267334,
    69.14342498779297,  1.8687255382537842,  0.9336621761322021,
    0.6543403267860413,
], dtype=float)

DEFAULT_PEAQ_STD = np.array([
    1.3240841627120972, 2.060269832611084,   129.22882080078125,
    145.09878540039062, 13.59294319152832,   33.13527297973633,
    0.9256784319877625, 1.41542387008667,    27.746591567993164,
    127.90895080566406, 3.1132972240448,     0.18070067465305328,
    0.39696332812309265,
], dtype=float)

LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(PEAQ_Score|ODG|DI|MOV\d+)(?![A-Za-z0-9])"
)


class Unsupported(Exception):
    pass


# ═══════════════════════════════════════════════════════════════
# UNIFIED MODEL
# ═══════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, c, norm):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), norm(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), norm(c),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class SEBlock(nn.Module):
    def __init__(self, c, hidden):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, hidden), nn.ReLU(),
            nn.Linear(hidden, c), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


def _build_backbone(sd):
    is_bn = any(k.endswith("running_mean") for k in sd if k.startswith("backbone."))
    norm = (lambda c: nn.BatchNorm2d(c)) if is_bn else (lambda c: nn.GroupNorm(1, c))

    idx = [int(m.group(1)) for k in sd for m in [re.match(r"backbone\.(\d+)\.", k)] if m]
    if not idx:
        raise Unsupported("checkpoint is not supported")
    layers, prev = [], None
    for i in range(max(idx) + 1):
        if f"backbone.{i}.block.0.weight" in sd:
            layers.append(ResBlock(sd[f"backbone.{i}.block.0.weight"].shape[0], norm))
            prev = "res"
        elif f"backbone.{i}.weight" in sd:
            w = sd[f"backbone.{i}.weight"]
            if w.dim() == 4:
                layers.append(nn.Conv2d(w.shape[1], w.shape[0], 3, padding=1))
                prev = "conv"
            else:
                layers.append(norm(w.shape[0]))
                prev = "norm"
        else:
            if prev == "norm":
                layers.append(nn.ReLU(inplace=True)); prev = "relu"
            elif prev == "res":
                layers.append(nn.MaxPool2d(2)); prev = "pool"
            else:
                raise Unsupported("checkpoint is not supported")
    if prev == "res":
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def _mlp(sd, prefix):
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    idx = sorted(int(m.group(1)) for k, v in sd.items()
                 for m in [pat.match(k)] if m and v.dim() == 2)
    if not idx:
        raise Unsupported("checkpoint is not supported")
    layers = []
    for j, i in enumerate(idx):
        if i != 3 * j:
            raise Unsupported("checkpoint is not supported")
        w = sd[f"{prefix}.{i}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if j < len(idx) - 1:
            layers += [nn.ReLU(inplace=True), nn.Dropout(0.0)]
    return nn.Sequential(*layers)


class UniversalModel(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.backbone = _build_backbone(sd)
        if "attention.se.2.weight" not in sd:
            raise Unsupported("checkpoint is not supported")
        h, c = sd["attention.se.2.weight"].shape
        self.attention = SEBlock(c, h)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if any(k.startswith("mov_head.") for k in sd):
            self.kind = "cascade"
            self.mov_head = _mlp(sd, "mov_head")
            self.di_head = _mlp(sd, "di_head")
            self.odg_head = _mlp(sd, "odg_head")
            self.n_out = 1 + 1 + sd["mov_head.3.weight"].shape[0]
        elif any(k.startswith("heads.") for k in sd):
            self.kind = "cerberus"
            ids = sorted({int(m.group(1)) for k in sd
                          for m in [re.match(r"heads\.(\d+)\.", k)] if m})
            self.heads = nn.ModuleList([_mlp(sd, f"heads.{i}") for i in ids])
            self.n_out = len(ids)
        elif any(k.startswith("head.") for k in sd):
            self.kind = "flat"
            self.head = _mlp(sd, "head")
            last = max(int(m.group(1)) for k in sd
                       for m in [re.match(r"head\.(\d+)\.weight$", k)] if m)
            self.n_out = sd[f"head.{last}.weight"].shape[0]
        else:
            raise Unsupported("checkpoint is not supported")

    def forward(self, x):
        z = self.pool(self.attention(self.backbone(x.unsqueeze(1)))).flatten(1)
        if self.kind == "cascade":
            mov = self.mov_head(z)
            di = self.di_head(torch.cat([z, mov], 1))
            odg = self.odg_head(torch.cat([z, mov, di], 1))
            return torch.cat([odg, di, mov], 1)
        if self.kind == "cerberus":
            return torch.cat([h(z) for h in self.heads], 1)
        return self.head(z)


# ═══════════════════════════════════════════════════════════════
# .ckpt LOADER
# ═══════════════════════════════════════════════════════════════
_KEEP = ("backbone.", "attention.", "head.", "heads.",
         "mov_head.", "di_head.", "odg_head.")


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    y_std = None
    sd = {}
    if isinstance(ck, dict) and "state_dict" in ck:
        for k, v in ck["state_dict"].items():
            for pre in ("model._orig_mod.", "model.", "_orig_mod."):
                if k.startswith(pre):
                    k = k[len(pre):]
                    break
            if k == "y_std":
                y_std = v.detach().float().cpu().numpy()
            else:
                sd[k] = v
    elif isinstance(ck, dict) and "feat" in ck and "head" in ck:
        arch = ck.get("arch")
        heads = ck["head"]
        if arch in ("flat", "single-metric"):
            prefix = "head."
        elif arch in ("cascade", "dynamic"):
            prefix = ""
        else:
            raise Unsupported("checkpoint is not supported")
        if len(heads) != 1:
            raise Unsupported("checkpoint is not supported")
        sd = dict(ck["feat"])
        for k, v in next(iter(heads.values())).items():
            sd[prefix + k] = v
    else:
        raise Unsupported("checkpoint is not supported")
    del ck
    return {k: v for k, v in sd.items() if k.startswith(_KEEP)}, y_std


def detect_label(path):
    tail = os.path.normpath(path).split(os.sep)[-3:]
    found = LABEL_RE.findall(" ".join(tail))
    return found[-1] if found else "PEAQ_Score"


class Entry:
    pass


def prepare_entry(path, name, device, args, norm_json):
    sd, y_std = load_checkpoint(path)
    model = UniversalModel(sd)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    e = Entry()
    e.name, e.model, e.n_out = name, model, model.n_out
    e.fallback_used = False

    if model.n_out == 13:
        e.mode = "peaq13"

        mean = DEFAULT_PEAQ_MEAN.copy()
        std  = DEFAULT_PEAQ_STD.copy()

        e.mean, e.std = mean, std
        # Always emit all 13 columns.
        e.cols = [(f"{name} [{PEAQ13[i]}]", PEAQ13[i], i) for i in range(13)]

    elif model.n_out == 1:
        label = args.single_target or detect_label(path)
        if label not in PEAQ13 and label != "PEAQ_Score":
            raise Unsupported("checkpoint is not supported")
        e.mode = "single"
        e.cols = [(f"{name} [{label}]", label, 0)]

    else:
        raise Unsupported("checkpoint is not supported")
    return e


# ═══════════════════════════════════════════════════════════════
# AUDIO
# ═══════════════════════════════════════════════════════════════
def _load_ffmpeg(path, sr):
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le",
           "-acodec", "pcm_f32le", "-ar", str(sr), "-ac", "1", "pipe:1"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 or not res.stdout:
        raise RuntimeError("ffmpeg could not read the file")
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def load_audio(path, sr):
    try:
        y, orig_sr = librosa.load(path, sr=None, mono=False)
        if y.ndim > 1:
            y = np.mean(y.astype(np.float64), axis=0)
        y = np.nan_to_num(y.astype(np.float32))
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        return y
    except Exception:
        return _load_ffmpeg(path, sr)


def make_windows(y, target_len, overlap):
    if len(y) <= target_len:
        if len(y) == target_len:
            return [y]
        return [np.pad(y, (0, target_len - len(y)), mode="constant")]
    step = max(1, int(target_len * (1.0 - overlap)))
    wins, start = [], 0
    while start + target_len <= len(y):
        wins.append(y[start:start + target_len])
        start += step
    if start < len(y):
        wins.append(y[-target_len:])
    return wins


def load_specs(path):
    y = load_audio(path, TARGET_SR)
    wins = make_windows(y, int(TARGET_SR * WINDOW_SEC), OVERLAP)
    specs = []
    for w in wins:
        S = np.abs(librosa.stft(w, n_fft=N_FFT, hop_length=HOP_LENGTH))
        # float16 — as in the training data; halves memory usage.
        specs.append(librosa.amplitude_to_db(S, ref=np.max).astype(np.float16))
    return specs


def iter_specs(files, workers):
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        it = iter(files)
        pending = deque()

        def submit():
            p = next(it, None)
            if p is not None:
                pending.append((p, ex.submit(load_specs, p)))

        for _ in range(workers + 1):
            submit()
        while pending:
            p, fut = pending.popleft()
            submit()
            try:
                yield p, fut.result(), None
            except Exception as exc:
                yield p, None, exc


@torch.inference_mode()
def predict_file(specs, entries, batch_size, device, amp):
    n = len(specs)
    acc = [np.zeros(e.n_out, dtype=np.float64) for e in entries]
    use_amp = amp and device == "cuda"
    for s in range(0, n, batch_size):
        batch = torch.from_numpy(np.stack(specs[s:s + batch_size])).to(device).float()
        for j, e in enumerate(entries):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = e.model(batch)
            acc[j] += out.float().sum(0).cpu().numpy()
    return [a / n for a in acc]


def find_audio(path):
    if os.path.isfile(path):
        return [path], os.path.dirname(os.path.abspath(path))
    out = []
    for r, _, fs in os.walk(path):
        for f in fs:
            if os.path.splitext(f)[1].lower() in AUDIO_EXT:
                out.append(os.path.join(r, f))
    return sorted(out), os.path.abspath(path)


def find_ckpts(root, include_last):
    if os.path.isfile(root):
        return [root], os.path.dirname(os.path.abspath(root))
    by_dir = {}
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".ckpt"):
                by_dir.setdefault(r, []).append(os.path.join(r, f))
    out = []
    for d in sorted(by_dir):
        ps = sorted(by_dir[d])
        best = [p for p in ps if not os.path.basename(p).lower().startswith("last")]
        out += ps if include_last else (best or ps)
    return out, os.path.abspath(root)


# ═══════════════════════════════════════════════════════════════
# GUI FALLBACK
# ═══════════════════════════════════════════════════════════════
def gui_fill(args):
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        sys.exit("tkinter is not available, use parameters: -i, -m, -o")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    if not args.input:
        is_dir = messagebox.askyesno(
            "Choose audio source",
            "Entire folder (subfolders are supported)?\n\n"
            "Yes — Folder\nNo — Single file",
        )
        if is_dir:
            args.input = filedialog.askdirectory(title="Folder with audio files")
        else:
            exts = " ".join("*" + e for e in sorted(AUDIO_EXT))
            args.input = filedialog.askopenfilename(
                title="Audio file",
                filetypes=[("Audio", exts), ("All files", "*.*")],
            )
    if not args.input:
        sys.exit("Nothing is chosen.")

    if not args.models:
        is_dir = messagebox.askyesno(
                    "Choose .ckpt source",
                    "Entire folder (subfolders are supported)?\n\n"
                    "Yes — Folder\nNo — Single .ckpt",
                )
        if is_dir:
            args.models = filedialog.askdirectory(title="Folder with models")
        else:
            exts = " ".join("*.ckpt")
            args.models = filedialog.askopenfilename(
                title=".ckpt file",
                filetypes=[("Model", exts), ("All files", "*.*")],
            )
    if not args.models:
        sys.exit("Nothing is chosen.")

    if not args.output:
        base = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
        args.output = filedialog.asksaveasfilename(
            title="Where to save results",
            defaultextension=".csv",
            initialdir=base,
            initialfile="peaq_results.csv",
            filetypes=[("CSV", "*.csv")],
        )
    if not args.output:
        sys.exit("Nothing is chosen.")
    root.destroy()
    args._gui = True


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Audio quality scoring")
    ap.add_argument("-i", "--input", help="audio file or folder")
    ap.add_argument("-m", "--models", help="folder with *.ckpt (or single .ckpt)")
    ap.add_argument("-o", "--output", help="CSV with results")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast on GPU (may be faster, but less precise)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--include-last", action="store_true",
                    help="include last*.ckpt in the search")
    ap.add_argument("--norm-json", default=None,
                    help="JSON with mean/std; overrides the built-in PEAQ stats")
    ap.add_argument("--odg-mean", type=float, default=None,
                    help="override the ODG mean (index 0)")
    ap.add_argument("--odg-std", type=float, default=None,
                    help="override the ODG std (index 0)")
    ap.add_argument("--single-target", default=None,
                    help="metric name for single-output checkpoints")
    args = ap.parse_args()
    args._gui = False

    if not (args.input and args.models and args.output):
        gui_fill(args)

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    batch_size = args.batch_size or (16 if device == "cuda" else 8)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | batch={batch_size}")

    files, in_root = find_audio(args.input)
    if not files:
        sys.exit("No audio found.")
    ckpts, m_root = find_ckpts(args.models, args.include_last)
    if not ckpts:
        sys.exit("No *.ckpt found.")
    print(f"Audios: {len(files)} | .ckpt files: {len(ckpts)}")

    norm_json = None
    if args.norm_json:
        with open(args.norm_json, "r", encoding="utf-8") as f:
            norm_json = json.load(f)

    # ---- Loading models ----
    entries, skipped = [], []
    for p in tqdm(ckpts, desc="Loading models", unit="mod."):
        name = (os.path.relpath(os.path.splitext(p)[0], m_root).replace("\\", "/")
                if os.path.isdir(args.models)
                else os.path.splitext(os.path.basename(p))[0])
        try:
            entries.append(prepare_entry(p, name, device, args, norm_json))
        except Exception as exc:
            skipped.append((p, str(exc)))
    for p, why in skipped:
        tqdm.write(f"[SKIP] {p}: {why}")
    if not entries:
        sys.exit("No supported models.")
    print(f"Models loaded: {len(entries)} (skipped: {len(skipped)})")

    columns = [c for e in entries for c in e.cols]
    col_names = [c[0] for c in columns]
    rows = {}

    def save():
        names = list(col_names)
        groups = {}
        for cn, metric, _ in columns:
            groups.setdefault(metric, []).append(cn)
        mean_cols = {f"MEAN [{m}]": cs for m, cs in groups.items() if len(cs) > 1}
        header = ["file"] + names + list(mean_cols)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            for fp in files:
                if fp not in rows:
                    continue
                r = rows[fp]
                vals = [r.get(c, float("nan")) for c in names]
                for cs in mean_cols.values():
                    v = np.array([r.get(c, np.nan) for c in cs], dtype=float)
                    vals.append(float(np.nanmean(v)) if np.isfinite(v).any()
                                else float("nan"))
                w.writerow([os.path.relpath(fp, in_root).replace("\\", "/")] + vals)

    # ---- Scoring ----
    try:
        bar = tqdm(total=len(files), desc="Scoring", unit="file")
        for path, specs, err in iter_specs(files, max(1, args.workers)):
            bar.set_postfix_str(os.path.basename(path)[:40])
            if err is not None:
                tqdm.write(f"[ERROR] {path}: {err}")
                rows[path] = {}
            else:
                means = predict_file(specs, entries, batch_size, device, args.amp)
                r = {}
                for e, m in zip(entries, means):
                    for cname, _, i in e.cols:
                        r[cname] = (float(m[i] * e.std[i] + e.mean[i])
                                    if e.mode == "peaq13" else float(m[i]))
                rows[path] = r
            bar.update(1)
        bar.close()
    except KeyboardInterrupt:
        print("\nInterrupted — saving what has been computed so far.")
    finally:
        save()

    print(f"\nDone. Results: {args.output}")

    # Final column summary.
    if col_names and rows:
        print(f"\nColumns in CSV ({len(col_names)}):")
        for cn in col_names[:12]:
            vals = np.array([rows[f].get(cn, np.nan) for f in rows], dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                print(f"  {cn}: mean={finite.mean():.4f}  std={finite.std():.4f}")
        if len(col_names) > 12:
            print(f"  ... and {len(col_names) - 12} more column(s) in CSV")

    if args._gui:
        try:
            from tkinter import messagebox
            messagebox.showinfo("Done", f"Results saved:\n{args.output}")
        except Exception:
            pass


if __name__ == "__main__":
    main()