import copy

import torch
import torch.nn as nn


class EMA:
    """
    Exponential moving average of model weights.

    DiT relies heavily on EMA for sample quality: we keep a shadow copy of the
    weights that is updated as ema = decay * ema + (1 - decay) * model each step,
    and sample from that copy rather than the raw training weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def save_checkpoint(path: str, model: nn.Module, ema: EMA, optimizer: torch.optim.Optimizer,
                    scheduler, epoch: int, step: int, args: dict) -> None:
    """
    Save model, EMA, optimizer, LR scheduler state and the run configuration.

    :param path: Destination file path.
    :param model: The DiT model being trained.
    :param ema: The EMA wrapper holding the shadow weights.
    :param optimizer: The optimizer whose state should be saved.
    :param scheduler: The LR scheduler whose state should be saved (so warmup/cosine resumes correctly).
    :param epoch: Current epoch index.
    :param step: Global training step.
    :param args: Parsed CLI args as a dict, for reproducible reloading/sampling.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": args,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}, step {step} -> {path}")


def load_checkpoint(path: str, model: nn.Module, ema: EMA = None,
                    optimizer: torch.optim.Optimizer = None, scheduler=None, map_location="cpu") -> dict:
    """
    Load a checkpoint in place. Returns the raw checkpoint dict (epoch, step, args).
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if ema is not None and "ema_state_dict" in checkpoint:
        ema.ema_model.load_state_dict(checkpoint["ema_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"Checkpoint loaded from {path} (epoch {checkpoint.get('epoch')}, step {checkpoint.get('step')})")
    return checkpoint
