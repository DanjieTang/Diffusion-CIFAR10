import argparse

import torch
from torchvision.utils import save_image

from checkpoint import load_checkpoint
from diffusion import GaussianDiffusion
from model import DiT


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from a trained DiT checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="generated.png")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Sample from EMA weights (recommended).")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")

    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--labels", type=int, nargs="+", default=None,
                        help="Explicit class ids to generate. Overrides --num_images.")
    parser.add_argument("--nrow", type=int, default=8)

    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = checkpoint["args"]

    num_classes = train_args.get("num_classes")
    if num_classes is None:
        # num_classes is inferred at train time; recover it from the embedding table size.
        state = checkpoint["ema_state_dict" if args.use_ema else "model_state_dict"]
        num_classes = state["label_embed.embedding_table.weight"].shape[0] - 1

    model = DiT(
        image_size=train_args["image_size"],
        image_channel=3,
        patch_size=train_args["patch_size"],
        hidden_size=train_args["hidden_size"],
        depth=train_args["depth"],
        num_heads=train_args["num_heads"],
        mlp_ratio=train_args["mlp_ratio"],
        num_classes=num_classes,
        class_dropout_prob=train_args["class_dropout_prob"],
    ).to(args.device)

    state_key = "ema_state_dict" if args.use_ema else "model_state_dict"
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    print(f"Loaded {state_key} from {args.checkpoint} (num_classes={num_classes})")

    diffusion = GaussianDiffusion(train_args["num_timesteps"], train_args["beta_schedule"]).to(args.device)

    if args.labels is not None:
        labels = torch.tensor(args.labels, device=args.device)
    else:
        labels = torch.randint(0, num_classes, (args.num_images,), device=args.device)

    images = diffusion.sample(
        model, labels.shape[0], labels, train_args["image_size"], image_channel=3,
        sampler=args.sampler, num_sampling_steps=args.num_sampling_steps,
        cfg_scale=args.cfg_scale, device=args.device,
    )
    images = (images.clamp(-1, 1) + 1) / 2
    save_image(images, args.output, nrow=args.nrow)
    print(f"Saved {labels.shape[0]} images -> {args.output}")


if __name__ == "__main__":
    main()
