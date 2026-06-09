import os
import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model, path)
            self.counter = 0

    @staticmethod
    def _save(model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)


def adjust_learning_rate(optimizer, epoch, args):
    """Cosine decay with optional warm-up (simple step schedule as fallback)."""
    if getattr(args, 'lradj', 'cosine') == 'cosine':
        min_lr = args.learning_rate * 0.01
        lr = min_lr + 0.5 * (args.learning_rate - min_lr) * (
            1 + np.cos(np.pi * epoch / args.train_epochs)
        )
    else:
        lr = args.learning_rate * (0.5 ** (epoch // 2))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
