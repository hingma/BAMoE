import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from data_provider.data_factory import data_provider
from models.BAMoE import BAMoE
from models.SingleBias import SingleBiasTransformer
from utils.metrics import metric, naive_forecast_scale
from utils.tools import EarlyStopping, adjust_learning_rate, count_parameters


MODEL_MAP = {
    'BAMoE': BAMoE,
    'SingleBias': SingleBiasTransformer,
}


class ExpForecast:
    """
    Unified training / evaluation loop for all forecasting experiments.

    Supports:
      - Experiment 1 (preliminary): SingleBias with bias_type ∈ {global, causal, local, periodic}
      - Experiment 2 (main):        BAMoE with heterogeneous experts
      - Experiment 3 (ablation):    BAMoE with varied routing, expert types, K
    """

    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _get_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            self._amp_device_type = 'cuda'
            return torch.device(f'cuda:{self.args.gpu}')
        if self.args.use_gpu and torch.backends.mps.is_available():
            self._amp_device_type = 'cpu'   # AMP not supported on MPS
            return torch.device('mps')
        self._amp_device_type = 'cpu'
        return torch.device('cpu')

    def _build_model(self):
        model_cls = MODEL_MAP[self.args.model]
        model = model_cls(self.args).to(self.device)
        n = count_parameters(model)
        print(f'Model: {self.args.model}  |  Parameters: {n:,}')
        return model

    def _checkpoint_path(self):
        tag = self.args.exp_name
        return os.path.join(self.args.checkpoints, tag, 'checkpoint.pth')

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        args = self.args
        ckpt_path = self._checkpoint_path()

        # Skip training if a checkpoint already exists and --resume is set
        if getattr(args, 'resume', False) and os.path.exists(ckpt_path):
            print(f'Checkpoint found, skipping training: {ckpt_path}')
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            # Still need the naive scale for MASE — load training data briefly
            train_set, _ = data_provider(args, 'train')
            self._naive_scale = naive_forecast_scale(train_set.data_x)
            return self

        train_set, train_loader = data_provider(args, 'train')
        val_set,   val_loader   = data_provider(args, 'val')
        self._naive_scale = naive_forecast_scale(train_set.data_x)
        early_stop = EarlyStopping(patience=args.patience)
        optimizer = Adam(self.model.parameters(), lr=args.learning_rate,
                         weight_decay=getattr(args, 'weight_decay', 1e-4))
        criterion = nn.MSELoss()
        use_amp = (self._amp_device_type == 'cuda') and getattr(args, 'use_amp', True)
        scaler  = GradScaler(self._amp_device_type, enabled=use_amp)

        for epoch in range(args.train_epochs):
            self.model.train()
            train_losses = []
            t0 = time.time()
            for x, y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.train_epochs}',
                              leave=False):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                with autocast(self._amp_device_type, enabled=use_amp):
                    pred, aux_loss, _ = self.model(x)
                    loss = criterion(pred, y) + aux_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(loss.item())

            val_loss = self._evaluate_loss(val_loader, criterion)
            lr = adjust_learning_rate(optimizer, epoch + 1, args)
            elapsed = time.time() - t0
            train_loss = np.mean(train_losses)
            print(f'Epoch {epoch + 1:3d} | train={train_loss:.4f} '
                  f'val={val_loss:.4f} | lr={lr:.2e} | {elapsed:.1f}s')

            if getattr(args, 'wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': lr,
                    'epoch_time_s': elapsed,
                })

            early_stop(val_loss, self.model, ckpt_path)
            if early_stop.early_stop:
                print('Early stopping.')
                break

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        return self

    # ------------------------------------------------------------------
    # Testing / evaluation
    # ------------------------------------------------------------------

    def test(self, load_ckpt=True):
        if load_ckpt:
            ckpt = self._checkpoint_path()
            if os.path.exists(ckpt):
                self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        use_amp = (self._amp_device_type == 'cuda') and getattr(self.args, 'use_amp', True)
        _, test_loader = data_provider(self.args, 'test')
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                with autocast(self._amp_device_type, enabled=use_amp):
                    pred, _, _ = self.model(x)
                preds.append(pred.cpu().numpy())
                trues.append(y.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        scale = getattr(self, '_naive_scale', None)
        mse_val, mae_val, crps_val, mase_val = metric(preds, trues, naive_scale=scale)
        print(f'Test  | MSE={mse_val:.4f}  MAE={mae_val:.4f}  '
              f'CRPS={crps_val:.4f}  MASE={mase_val:.4f}')

        if getattr(self.args, 'wandb', False):
            wandb.log({
                'test_mse': mse_val,
                'test_mae': mae_val,
                'test_crps': crps_val,
                'test_mase': mase_val,
            })

        self._save_results(mse_val, mae_val, crps_val, mase_val)
        return mse_val, mae_val, crps_val, mase_val

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_loss(self, loader, criterion):
        use_amp = (self._amp_device_type == 'cuda') and getattr(self.args, 'use_amp', True)
        self.model.eval()
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                with autocast(self._amp_device_type, enabled=use_amp):
                    pred, aux_loss, _ = self.model(x)
                losses.append((criterion(pred, y) + aux_loss).item())
        self.model.train()
        return np.mean(losses)

    def _save_results(self, mse_val, mae_val, crps_val, mase_val):
        args = self.args
        out_dir = os.path.join(args.results, args.exp_name)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(args.results, 'summary.csv')
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['exp_name', 'model', 'data', 'pred_len',
                                 'mse', 'mae', 'crps', 'mase'])
            writer.writerow([args.exp_name, args.model, args.data, args.pred_len,
                             f'{mse_val:.4f}', f'{mae_val:.4f}',
                             f'{crps_val:.4f}', f'{mase_val:.4f}'])
