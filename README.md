# Pixel-space DiT — CIFAR10 & ImageNet

A class-conditional **Diffusion Transformer (DiT)** trained from scratch in pixel space
(no pretrained VAE, no pretrained anything). DDPM epsilon-prediction with classifier-free
guidance, adaLN-Zero conditioning, and EMA weights. Supports CIFAR10 and a local ImageNet
folder, driven entirely by command-line arguments with a YAML grid-sweep runner.

## Project layout

| File | Purpose |
|------|---------|
| `model.py` | The DiT: patch embed, timestep/label embedders, adaLN-Zero blocks, sin-cos pos embed. |
| `diffusion.py` | `GaussianDiffusion`: beta schedules, `q_sample`, training loss, DDPM + DDIM sampling with CFG. |
| `data.py` | DataLoaders for CIFAR10 and local ImageNet (`ImageFolder`). |
| `checkpoint.py` | EMA wrapper + checkpoint save/load. |
| `train.py` | Argparse training entry point (W&B optional, EMA, preview samples, checkpoints). |
| `sample.py` | Generate images from a trained checkpoint. |
| `run_sweep.py` | Run `train.py` over every combination in a sweep YAML, resumable. |
| `sweep_config.yaml` | Example sweep definition. |
| `legacy_code/` | The original Jupyter-notebook implementations, kept for reference. |

## Setup (uv)

```bash
uv sync                      # create .venv and install from pyproject.toml
```

Run anything with `uv run python ...` (no manual venv activation needed).

## Train

CIFAR10 (32×32):

```bash
uv run python train.py --dataset cifar10 --image_size 32 --patch_size 2 \
    --hidden_size 384 --depth 12 --num_heads 6 --batch_size 128 --epochs 100
```

ImageNet from a local folder (`ImagenetHighResolution/<class>/<image>.jpeg`):

```bash
uv run python train.py --dataset imagenet --imagenet_path ./ImagenetHighResolution \
    --image_size 64 --patch_size 4 --hidden_size 768 --depth 12 --num_heads 12 \
    --batch_size 64 --epochs 100
```

`num_classes` is inferred automatically (10 for CIFAR10, or one per sub-folder for ImageNet).
Preview grids and checkpoints land in `--output_dir` (default `./runs`). Resume with
`--resume runs/dit_cifar10_epoch_50.pth`.

Enable Weights & Biases logging by passing both `--project` and `--entity`.

## Sample

```bash
uv run python sample.py --checkpoint runs/dit_cifar10_epoch_100.pth \
    --labels 0 1 2 3 4 5 6 7 8 9 --cfg_scale 4.0 --sampler ddim --num_sampling_steps 250 \
    --output samples.png
```

## Hyperparameter sweeps

Edit `sweep_config.yaml`: scalars are held fixed, lists become sweep dimensions expanded
as a full grid. Every key must match a `--flag` in `train.py` (validated before running).

```bash
uv run python run_sweep.py --config sweep_config.yaml          # run the grid
uv run python run_sweep.py --config sweep_config.yaml --dry-run # print commands only
```

Progress is tracked in `sweep_progress.json`, so an interrupted sweep skips completed runs
on the next invocation. Use `--continue-on-error` to keep going past a failed run and
`--force-rerun` to ignore prior progress.

## Key design choices

- **Pixel-space**, so the model patchifies raw RGB images directly — no VAE.
- **adaLN-Zero** conditioning: timestep + class embeddings drive per-block shift/scale/gate,
  initialized to zero so each block starts as identity.
- **Classifier-free guidance**: a fraction (`--class_dropout_prob`) of labels are dropped to a
  null token during training; `--cfg_scale > 1` mixes conditional/unconditional predictions at
  sampling time.
- **EMA** weights are used for all sampling.
