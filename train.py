import argparse
import datetime
import os
import uuid

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.utils import save_image
from tqdm import tqdm

import wandb

from checkpoint import EMA, load_checkpoint, save_checkpoint
from data import prepare_dataset
from diffusion import GaussianDiffusion
from model import DiT


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a pixel-space class-conditional DiT.")

    # Dataset & Paths
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument("--data_root", type=str, default="./data", help="CIFAR10 download/cache dir.")
    parser.add_argument("--imagenet_path", type=str, default="./ImagenetHighResolution",
                        help="Root of local ImageNet class folders (ImageFolder layout).")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--min_image_size", type=int, default=0,
                        help="Drop ImageNet images whose shorter edge is below this (0 disables). "
                             "Avoids training on upscaled, blurry images. No effect on CIFAR10.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./train_log",
                        help="Root log folder. Each training run gets its own timestamped subfolder.")

    # Model Architecture
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--class_dropout_prob", type=float, default=0.1,
                        help="Probability of dropping the label for classifier-free guidance.")

    # Diffusion
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--num_sampling_steps", type=int, default=50, help="Steps for the DDIM sampler.")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Guidance scale used for preview samples.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["constant", "cosine"],
                        help="Post-warmup LR schedule. 'constant' holds the peak LR; 'cosine' decays it.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear warmup steps before the main schedule takes over.")
    parser.add_argument("--lr_min", type=float, default=0.0,
                        help="Floor LR the cosine schedule decays to (eta_min). Ignored for 'constant'.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")

    # Logging & Checkpointing
    parser.add_argument("--log_every", type=int, default=50, help="Log training loss every N steps.")
    parser.add_argument("--sample_every", type=int, default=1, help="Generate preview samples every N epochs.")
    parser.add_argument("--ckpt_every", type=int, default=5, help="Save a checkpoint every N epochs.")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def build_scheduler(optimizer, args, total_steps):
    """
    Linear warmup followed by the chosen main schedule, stepped once per iteration.

    The warmup ramps the LR from 1% of the peak up to the peak over `warmup_steps`;
    afterwards 'cosine' anneals down to `lr_min` over the remaining steps, while
    'constant' simply holds the peak LR.
    """
    warmup_steps = max(0, min(args.warmup_steps, total_steps))
    decay_steps = max(1, total_steps - warmup_steps)

    if args.lr_scheduler == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=args.lr_min)
    else:  # constant: hold the peak LR (factor 1.0 leaves the LR unchanged)
        main = ConstantLR(optimizer, factor=1.0, total_iters=decay_steps)

    if warmup_steps == 0:
        return main
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    return SequentialLR(optimizer, [warmup, main], milestones=[warmup_steps])


@torch.no_grad()
def sample_preview(diffusion, ema_model, num_classes, args, path):
    """Sample one image per class (capped at 10) from the EMA weights and save a grid."""
    n = min(num_classes, 10)
    labels = torch.arange(n, device=args.device)
    images = diffusion.sample(
        ema_model, n, labels, args.image_size, image_channel=3,
        sampler=args.sampler, num_sampling_steps=args.num_sampling_steps,
        cfg_scale=args.cfg_scale, device=args.device,
    )
    images = (images.clamp(-1, 1) + 1) / 2  # back to [0, 1] for saving
    save_image(images, path, nrow=n)
    return images


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Each run (including each sweep trial) gets its own subfolder under the log root.
    # The timestamp + short random suffix keep concurrent sweep trials from colliding.
    run_tag = args.run_name or args.dataset
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{run_tag}_{stamp}_{uuid.uuid4().hex[:6]}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    use_wandb = args.project is not None and args.entity is not None
    if use_wandb:
        run = wandb.init(entity=args.entity, project=args.project, name=args.run_name, config=vars(args))

    train_loader, num_classes = prepare_dataset(
        args.dataset, args.data_root, args.imagenet_path,
        args.image_size, args.batch_size, args.num_workers,
        min_image_size=args.min_image_size,
    )
    print(f"Dataset: {args.dataset} | classes: {num_classes} | batches/epoch: {len(train_loader)}")

    model = DiT(
        image_size=args.image_size,
        image_channel=3,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=num_classes,
        class_dropout_prob=args.class_dropout_prob,
    ).to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = GaussianDiffusion(args.num_timesteps, args.beta_schedule).to(args.device)
    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, args, total_steps)

    start_epoch, global_step = 0, 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, ema, optimizer, scheduler, map_location=args.device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, labels in progress:
            images = images.to(args.device)
            labels = labels.to(args.device)

            loss = diffusion.training_loss(model, images, labels)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if use_wandb and global_step % args.log_every == 0:
                run.log({"train_loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=global_step)

        if (epoch + 1) % args.sample_every == 0:
            sample_path = os.path.join(run_dir, f"samples_epoch_{epoch + 1}.png")
            images = sample_preview(diffusion, ema.ema_model, num_classes, args, sample_path)
            print(f"Saved preview samples -> {sample_path}")
            if use_wandb:
                run.log({"samples": wandb.Image(images)}, step=global_step)

        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            ckpt_path = os.path.join(run_dir, f"dit_{args.dataset}_epoch_{epoch + 1}.pth")
            save_checkpoint(ckpt_path, model, ema, optimizer, scheduler, epoch, global_step, vars(args))

    if use_wandb:
        run.finish()


if __name__ == "__main__":
    main()
