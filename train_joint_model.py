import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

import h5py
import kornia.augmentation
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from mrpro.nn.nets import UNet
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure

if not sys.warnoptions:
    print('setting warnings to ignore')
    import os
    import warnings

    warnings.simplefilter('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'


class BatchDict(TypedDict):
    localizer: torch.Tensor
    b1: torch.Tensor
    mask: torch.Tensor
    location: torch.Tensor
    orientation: torch.Tensor
    subject_idx: torch.Tensor
    slice_idx: torch.Tensor


locations = ('Minnesota', 'Buch', 'Heidelberg')
orientations = ('Sagittal', 'Coronal', 'Transversal')


def complex_to_real(x: torch.Tensor) -> torch.Tensor:
    return rearrange(torch.view_as_real(x), 'b c ... realimag -> b (c realimag) ...')


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(rearrange(x, 'b (c realimag) ... -> b c ... realimag', realimag=2).contiguous())


class B1LocalizerDS(torch.utils.data.Dataset):
    def pad_or_random_crop(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n_dims = len(self.size)
        shapes = [t.shape[-n_dims:] for t in tensors]
        if any(s != shapes[0] for s in shapes):
            raise ValueError('All tensors must have the same spatial dimensions')
        pads = []
        for size_dim, shape_dim in zip(reversed(self.size), reversed(shapes[0]), strict=False):
            diff = size_dim - shape_dim
            if diff < 0:
                start = torch.randint(0, -diff + 1, (1,)).item()
                pads.extend([-start, diff + start])
            else:
                pads.extend([diff // 2, diff - diff // 2])
        return tuple(torch.nn.functional.pad(t, tuple(pads), mode='replicate') for t in tensors)

    def __init__(
        self,
        data_path: Path,
        orientations: Sequence[Literal['Sagittal', 'Coronal', 'Transversal']] = orientations,
        locations: Sequence[Literal['Minnesota', 'Buch', 'Heidelberg']] = locations,
        fold: int = 0,
        n_folds: int = 4,
        train: bool = True,
    ):
        self.data_path = data_path
        self.size = (96, 128)

        all_files = []
        for fn in sorted(Path(data_path).rglob('*.h5')):
            with h5py.File(fn, 'r') as f:
                location = f.attrs['location']
                orientation = f.attrs['orientation']
                subject_idx = int(f.attrs['index'])
                if location in locations and orientation in orientations:
                    all_files.append((fn, subject_idx))

        if n_folds < 2:
            raise ValueError('n_folds must be greater than 1')
        if not 0 <= fold < n_folds:
            raise ValueError(f'fold must be in [0, {n_folds - 1}], got {fold}')
        if train:
            files_to_use = [fn for (fn, subject_idx) in all_files if subject_idx % n_folds != fold]
        else:
            files_to_use = [fn for (fn, subject_idx) in all_files if subject_idx % n_folds == fold]

        files = []
        n_slices = []
        for fn in files_to_use:
            with h5py.File(fn, 'r') as f:
                files.append(fn)
                n_slices.append(f['b1'].shape[0])

        if not files:
            raise ValueError(f'No files found for locations {locations} and orientations {orientations} in {data_path}')
        self.files = files
        self.n_slices = n_slices

    def __len__(self):
        return sum(self.n_slices)

    def __getitem__(self, idx):
        cum_slices = torch.cumsum(torch.tensor(self.n_slices), dim=0)
        total_slices = cum_slices[-1]
        if not -total_slices <= idx < total_slices:
            raise IndexError(f'Index {idx} is out of bounds for dataset of length {total_slices}')
        idx = idx % total_slices
        file_idx = torch.searchsorted(cum_slices, idx, side='right')
        slice_idx = idx - cum_slices[file_idx - 1] if file_idx > 0 else idx

        with h5py.File(self.files[file_idx], 'r') as f:
            b1 = torch.from_numpy(f['b1'][slice_idx].astype(np.complex64)).moveaxis(-1, 0)
            mask = torch.from_numpy(f['mask'][slice_idx].astype(np.float32))[None, ...]
            localizer = torch.from_numpy(f['localizer'][slice_idx].astype(np.complex64)).moveaxis(-1, 0)
            location = locations.index(f.attrs['location'])
            orientation = orientations.index(f.attrs['orientation'])
            subject_idx = f.attrs['index']

        b1, mask, localizer = self.pad_or_random_crop(b1, mask, localizer)

        return {
            'localizer': localizer,
            'b1': b1,
            'mask': mask,
            'location': torch.tensor(location, dtype=torch.long),
            'orientation': torch.tensor(orientation, dtype=torch.long),
            'subject_idx': torch.tensor(subject_idx, dtype=torch.long),
            'slice_idx': slice_idx.long(),
        }


class B1LocalizerModule(pl.LightningDataModule):
    """DataModule providing augmented inputs and targets."""

    def __init__(
        self,
        data_dir: str = '/echo/allgemein/projects/MRpro/B1/FFZHK/processed/',
        batch_size: int = 8,
        num_workers: int = 8,
        fold: int = 0,
        n_folds: int = 4,
        orientations: Sequence[Literal['Sagittal', 'Coronal', 'Transversal']] = ('Sagittal', 'Coronal', 'Transversal'),
        locations: Sequence[Literal['Minnesota', 'Buch', 'Heidelberg']] = ('Buch', 'Minnesota', 'Heidelberg'),
        p_affine: float = 0.75,
        affine_degrees: float = 5,
        affine_translate: float = 0.02,
        affine_scale: float = 0.02,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.orientations = orientations
        self.locations = locations
        self.fold = fold
        self.n_folds = n_folds
        self.affine = kornia.augmentation.RandomAffine(
            p=p_affine,
            degrees=affine_degrees,
            translate=(affine_translate, affine_translate),
            scale=(1 - affine_scale, 1 + affine_scale),
        )

    def setup(self, stage: str | None = None) -> None:
        if stage != 'fit':
            raise NotImplementedError(f'not implemeted yet: {stage}')
        self.train_dataset = B1LocalizerDS(
            self.data_dir,
            self.orientations,
            self.locations,
            fold=self.fold,
            n_folds=self.n_folds,
            train=True,
        )
        self.val_dataset = B1LocalizerDS(
            self.data_dir,
            self.orientations,
            self.locations,
            fold=self.fold,
            n_folds=self.n_folds,
            train=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)

    def on_after_batch_transfer(self, batch: BatchDict, dataloader_idx: int):
        if self.trainer.training:
            params = self.affine.forward_parameters(batch['mask'].shape)
            mask = self.affine(batch['mask'], params=params)
            localizer = real_to_complex(self.affine(complex_to_real(batch['localizer']), params=params))
            b1 = real_to_complex(self.affine(complex_to_real(batch['b1']), params=params))
            batch.update(localizer=localizer, b1=b1, mask=mask)
        return batch


class MaskedL1Loss(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        loss = (mask * (pred - target).abs()).sum()
        loss = loss / (mask.sum() + 1e-9)
        return loss


class MaskedMSELoss(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        loss = (mask * (pred - target).abs().square()).sum()
        loss = loss / (mask.sum() + 1e-9)
        return loss


class B1Predictor(pl.LightningModule):
    """B1 prediction from localizer."""

    def __init__(
        self,
        lr: float = 8e-4,
        weight_decay: float = 1e-3,
        n_features: tuple[int, ...] = (64, 128, 192, 192),
        attention_depths: tuple[int, ...] = (-1, -2),
        append_rss: bool = True,
        n_rx: int = 32,
        n_tx: int = 8,
        embedding_dim: int = 128,
        loss: torch.nn.Module = MaskedMSELoss(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['loss'])

        self.unet = UNet(
            n_dim=2,
            n_channels_in=n_rx * 2 + append_rss,
            n_channels_out=n_tx * 2,
            cond_dim=embedding_dim,
            n_features=n_features,
            attention_depths=attention_depths,
            encoder_blocks_per_scale=2,
        )
        self.unet = torch.compile(self.unet, dynamic=False, fullgraph=True)
        self.location_embedding = torch.nn.Embedding(len(locations), embedding_dim // 2)
        self.orientation_embedding = torch.nn.Embedding(len(orientations), embedding_dim // 2)
        self.cond_mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.Dropout(0.1),
            torch.nn.GELU(),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.Dropout(0.1),
        )
        self.criterion = loss
        self.validation_plotted = set()

    def forward(
        self,
        localizer: torch.Tensor,
        location: torch.Tensor,
        orientation: torch.Tensor,
    ) -> torch.Tensor:
        localizer_real = complex_to_real(localizer)
        if self.hparams.append_rss:
            rss = localizer.abs().square().sum(dim=1).sqrt()
            localizer_real = torch.cat([localizer_real, rss[:, None]], dim=1)

        location_emb = self.location_embedding(location)
        orientation_emb = self.orientation_embedding(orientation)
        cond = torch.cat([location_emb, orientation_emb], dim=1)
        cond = self.cond_mlp(cond)

        prediction = self.unet(localizer_real, cond=cond)
        prediction = real_to_complex(prediction).to(torch.complex64)
        return prediction

    def training_step(self, batch: BatchDict, batch_idx: int):
        pred_complex = self(batch['localizer'], batch['location'], batch['orientation'])
        loss = self.criterion(pred_complex, batch['b1'], batch['mask'])
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: BatchDict, batch_idx: int):
        prediction = self(batch['localizer'], batch['location'], batch['orientation'])
        gt = batch['b1']
        mask = batch['mask']
        mask_sum = mask.sum() + 1e-9

        location_str = locations[batch['location'][0].item()]
        orientation_str = orientations[batch['orientation'][0].item()]
        metrics = {}

        metrics['loss'] = self.criterion(prediction, gt, mask)
        metrics['l1'] = (mask * (prediction - gt).abs()).sum() / (mask_sum * gt.shape[1])
        metrics['mse_complex'] = (mask * (prediction - gt).abs()).square().sum() / (mask_sum * gt.shape[1])
        metrics['ssim_magnitude'] = structural_similarity_index_measure(
            prediction.abs() * mask, gt.abs() * mask, data_range=1.0
        )
        angle_diff = prediction.angle() - gt.angle()
        angle_error = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)).abs()
        metrics['mae_phase'] = (angle_error * mask).sum() / (mask_sum * gt.shape[1])

        for name, value in metrics.items():
            self.log(f'validation/{name}', value, sync_dist=True, prog_bar=name == 'loss')
            self.log(f'validation/{location_str}/{orientation_str}/{name}', value, sync_dist=True)

        subject_idx = batch['subject_idx'][0].item()
        slice_idx = batch['slice_idx'][0].item()
        orientation_idx = batch['orientation'][0].item()
        location_idx = batch['location'][0].item()

        if slice_idx == 5 and (orientation_idx, location_idx) not in self.validation_plotted:
            self.validation_plotted.add((orientation_idx, location_idx))
            n_tx = self.hparams.n_tx
            fig, axes = plt.subplots(4, n_tx, figsize=(n_tx * 2, 8), constrained_layout=True)
            nan_mask = np.where(mask[0].cpu().numpy() < 0.5, np.nan, 1)
            gt_mag = gt.abs()[0].cpu().numpy() * nan_mask
            pred_mag = prediction.abs()[0].cpu().numpy() * nan_mask
            gt_phase = gt.angle()[0].cpu().numpy() * nan_mask
            pred_phase = prediction.angle()[0].cpu().numpy() * nan_mask

            for i in range(n_tx):
                axes[0, i].imshow(gt_mag[i], cmap='gray')
                axes[1, i].imshow(pred_mag[i], cmap='gray')
                axes[2, i].imshow(gt_phase[i], cmap='hsv', vmin=-np.pi, vmax=np.pi)
                axes[3, i].imshow(pred_phase[i], cmap='hsv', vmin=-np.pi, vmax=np.pi)
                axes[0, i].set_title(f'Coil {i + 1}')
            axes[0, 0].set_ylabel('GT Mag')
            axes[1, 0].set_ylabel('Pred Mag')
            axes[2, 0].set_ylabel('GT Phase')
            axes[3, 0].set_ylabel('Pred Phase')

            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(f'S:{subject_idx} Sl:{slice_idx} L:{location_str} O:{orientation_str}')
            self.logger.run[f'val_images/{location_str}/{orientation_str}'].log(fig)
            plt.close(fig)

        return metrics['loss']

    def on_validation_epoch_end(self):
        self.validation_plotted.clear()

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if any(key in name for key in ('embedding', 'last_block', 'first')):
                no_decay.append(param)
            else:
                decay.append(param)

        opt_g = torch.optim.AdamW(
            [
                {'params': decay, 'weight_decay': self.hparams.weight_decay},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=self.hparams.lr,
        )

        sched_g = torch.optim.lr_scheduler.OneCycleLR(
            opt_g,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.2,
            final_div_factor=20,
            div_factor=25,
        )
        return [opt_g], [{'scheduler': sched_g, 'interval': 'step'}]


class MyLightningCLI(LightningCLI):
    """Custom CLI to inject logger and callbacks with minimal arguments."""

    def add_arguments_to_parser(self, parser):
        parser.add_argument('--experiment', type=str, default='B1 prediction', help='Name of the experiment.')
        parser.add_argument('--neptune_project', type=str, default='ptb/b1p', help='Neptune project name.')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/b1', help='path for checkpoints.')
        parser.set_defaults({'data': {'class_path': '__main__.B1LocalizerModule'}})

    def before_instantiate_classes(self):
        config = self.config.fit
        fold = config.data.init_args.fold
        experiment_name = config.experiment

        logger = NeptuneLogger(
            project=config.neptune_project,
            name=f'{experiment_name}-fold{fold}',
            tags=[experiment_name, f'fold_{fold}'],
            log_model_checkpoints=False,
            dependencies='infer',
            source_files=[__file__],
        )
        _ = logger.run

        checkpoint_callback = ModelCheckpoint(
            monitor='validation/loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            dirpath=f'{config.checkpoint_dir}/{logger.version}',
            filename=f'{logger.version}-{{epoch:02d}}-{{val_loss:.4f}}',
        )

        config.trainer.logger = logger
        config.trainer.callbacks = [
            LearningRateMonitor(logging_interval='step'),
            checkpoint_callback,
        ]

    def after_instantiate_classes(self):
        """Log model summary after the model and logger are instantiated."""
        self.trainer.logger.log_model_summary(model=self.model, max_depth=-1)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch._inductor.config.worker_start_method = 'fork'
    torch._inductor.config.compile_threads = 4
    torch._dynamo.config.capture_scalar_outputs = True

    cli = MyLightningCLI(
        B1Predictor,
        B1LocalizerModule,
        save_config_callback=None,
        subclass_mode_data=True,
        seed_everything_default=123,
        trainer_defaults={
            'max_epochs': 400,
            'accelerator': 'gpu',
            'devices': 1,
            'precision': '32-true',
            'log_every_n_steps': 2,
            'check_val_every_n_epoch': 4,
        },
    )
