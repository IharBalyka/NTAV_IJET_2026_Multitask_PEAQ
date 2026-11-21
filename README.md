# Non-Intrusive PEAQ Approximation Using Deep Neural Networks: a Comparison of Single-Task and Multitask Approaches

Official repository for the paper **“Non-Intrusive PEAQ Approximation Using Deep Neural Networks: a Comparison of Single-Task and Multitask Approaches”** (IJET, 2026).

This repository contains PyTorch checkpoints for four PEAQ‑approximation architectures across eight model sizes, together with an inference script (`peaq_predict.py`) and architecture diagrams. The models estimate the Objective Difference Grade (ODG) and, depending on the architecture, also the Distortion Index (DI) and the eleven Model Output Variables (MOV1–MOV11) directly from a degraded audio signal—without access to the reference recording.

---

## Publication

> Balyka, I., Zieliński, S.K. (202x).  
> **Non-Intrusive PEAQ Approximation Using Deep Neural Networks: a Comparison of Single-Task and Multitask Approaches.**  
> *International Journal of Electronics and Telecommunications (IJET)*, vol. xx, no. x, pp. xxx–xxx.  
> DOI: `xxxxx` *(will be updated upon final publication)*

If you use this code or the provided checkpoints, please cite the paper above.

### BibTeX

```bibtex
@article{balyka2026nonintrusive,
  author  = {Balyka, Ihar and Zieli{\'n}ski, S{\l}awomir K.},
  title   = {Non-Intrusive {PEAQ} Approximation Using Deep Neural Networks:
             a Comparison of Single-Task and Multitask Approaches},
  journal = {International Journal of Electronics and Telecommunications},
  year    = {202x},
  volume  = {xx},
  number  = {x},
  pages   = {xxx--xxx},
  doi     = {xxxxx}
}
```

---

## Architectures

Four architectures are compared in the paper:

| Architecture | `arch` tag | Description | Output |
|---|---|---|---|
| **Single-task** | `single` | CNN feature extractor followed by one regression head for ODG only. | ODG |
| **Multitask-DI** | `multitask` | Shared CNN features fed into three independent heads: MOVs (11), DI, and ODG. | ODG, DI, MOV1–MOV11 |
| **Multitask-CAS** | `cascade` | Cascade architecture that mirrors the PEAQ pipeline: features → MOVs → DI → ODG. Fixed loss weights. | ODG, DI, MOV1–MOV11 |
| **Multitask-DYN** | `dynamic` | Same cascade topology as `cascade`, but with dynamic loss weights that gradually shift focus from MOVs to DI and finally to ODG. | ODG, DI, MOV1–MOV11 |

All models share the same convolutional feature extractor and the same input representation (log‑mel spectrograms, 7‑second windows, 44.1 kHz, `n_fft=2048`, `hop_length=256`).  
For each audio file, the final prediction is the average of the model outputs over all 7‑second windows (50 % overlap).

Architecture diagrams for each model and size are available as `Overview.pdf` inside every checkpoint folder.

---

## Model checkpoints

The repository provides checkpoints for **eight model sizes** (parameter counts are approximate):

| Size tag | Parameters | Single-task | Multitask-DI | Multitask-CAS | Multitask-DYN |
|---|---|---|---|---|---|
| `0.0039M` | ≈ 3.9 k | ✅ | ✅ | ✅ | ✅ |
| `0.0078M` | ≈ 7.8 k | ✅ | ✅ | ✅ | ✅ |
| `0.0156M` | ≈ 15.6 k | ✅ | ✅ | ✅ | ✅ |
| `0.0312M` | ≈ 31.2 k | ✅ | ✅ | ✅ | ✅ |
| `0.5M`     | ≈ 0.5 M | ✅ | ✅ | — | ✅ |
| `2M`       | ≈ 2 M   | ✅ | ✅ | — | — |
| `4M`       | ≈ 4 M   | ✅ | ✅ | ✅ | ✅ |
| `8M`       | ≈ 8 M   | ✅ | ✅ | — | — |

`—` means the architecture was not trained for that size (as described in the paper).  

Due to Github limitations bigger checkpoints are available at Kaggle: www.kaggle.com/datasets/igorewka/ntav-ijet-2026-multitask-peaq

---

## Installation

**Clone the repository**

```bash
git clone https://github.com/IharBalyka/IJET_2026_Multitask_PEAQ.git
cd IJET_2026_Multitask_PEAQ
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

Required libraries: `torch`, `librosa`, `numpy`, `tqdm`, and `ffmpeg` (system dependency, used as a fallback audio loader).

---

## Usage

The main entry point is `peaq_predict.py`. It can be run with a simple GUI (no arguments) or from the command line.

### GUI mode

```bash
python peaq_predict.py
```

The script will ask for:
- an audio file or a folder with audio files,
- a `.ckpt` checkpoint or a folder with `.ckpt` checkpoints,
- a path for the output CSV.

### Command-line mode

```bash
python peaq_predict.py -i path/to/audio -m path/to/checkpoints -o results.csv
```

#### Arguments

| Argument | Description | Default |
|---|---|---|
| `-i`, `--input` | Audio file or folder (subfolders are searched recursively). | *required* (unless GUI) |
| `-m`, `--models` | Folder with `.ckpt` files or a single `.ckpt` file. | *required* (unless GUI) |
| `-o`, `--output` | Path to the output CSV file. | *required* (unless GUI) |
| `--batch-size` | Batch size for inference. | `16` on GPU, `8` on CPU |
| `--workers` | Number of parallel audio‑loading threads. | `min(8, cpu_count)` |
| `--amp` | Enable fp16 autocast on GPU (faster, slightly less precise). | `False` |
| `--cpu` | Force CPU inference. | `False` |

---

## Output format

The output is a CSV file with one row per audio file and one column per model‑metric combination.

- For **Single‑task** models: one column, e.g., `Single-task_0.0039M [ODG]`.
- For **Multitask / Cascade / Dynamic** models: 13 columns, e.g., `Multitask-DI_0.0039M [ODG]`, `Multitask-DI_0.0039M [DI]`, `Multitask-DI_0.0039M [MOV1]`, …, `Multitask-DI_0.0039M [MOV11]`.

If the same metric is produced by more than one model, an additional `MEAN[<metric>]` column is appended, containing the average across those models.

All values are de‑standardised back to the original PEAQ scale using the built‑in PEAQ z‑score statistics (`PEAQ_MEAN` / `PEAQ_STD`).

**Example (excerpt):**

```csv
file,Single-task_0.0039M [ODG],Multitask-DI_0.0039M [ODG],Multitask-DI_0.0039M [DI],...,MEAN[ODG]
audio/sample_01.wav,-0.42,-0.39,1.87,...,-0.405
audio/sample_02.wav,-1.15,-1.08,3.22,...,-1.115
```

---

## Audio requirements

- **Formats:** WAV, MP3, OGG, OPUS, MP4, M4A, AAC, FLAC.
- **Sample rate:** any; internally resampled to **44.1 kHz**.
- **Channels:** mono or stereo; stereo is averaged to mono.
- **Length:** any. Files shorter than 7 seconds are zero‑padded; longer files are split into 7‑second windows with 50 % overlap, and the final score is the mean over all windows.

---

Each checkpoint folder contains:
- `*.ckpt` — training runs.
- `Overview.pdf` — architecture diagram for that specific size and architecture.

## License

The code and model checkpoints are released for research and non‑commercial use.  
If you use them in academic work, please cite the paper as shown above.

---

## Contact

For questions about the code or the models, please open an issue on GitHub or contact:

- **Ihar Balyka** — `ihar.balyka@pb.edu.pl`

Faculty of Computer Science, Białystok University of Technology, Poland.
