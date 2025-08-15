import warnings
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
from neptune.common.warnings import NeptuneUnsupportedValue
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure

warnings.filterwarnings('error', category=NeptuneUnsupportedValue)


class BatchDict(TypedDict):
    localizer: torch.Tensor
    b1: torch.Tensor
    mask: torch.Tensor
    location: torch.Tensor
    orientation: torch.Tensor
    subject_idx: torch.Tensor
    slice_idx: torch.Tensor


class DataAugmentation(torch.nn.Module):
    """Self-contained data augmentation module for Stage 2."""

    def __init__(
        self,
        p_affine: float = 0.5,
        degrees: float = 5,
        translate: float = 0.01,
        scale: float = 0.02,
    ):
        super().__init__()
        self.affine = kornia.augmentation.RandomAffine(
            p=p_affine,
            degrees=degrees,
            translate=(translate, translate),
            scale=(1 - scale, 1 + scale),
        )

    def forward(self, *tensors):
        params = self.affine.forward_parameters(tensors[0].shape)
        tensors = tuple(self.affine(t, params=params) for t in tensors)
        return tensors


locations = ('Minnesota', 'Buch', 'Heidelberg')
orientations = ('Sagittal', 'Coronal', 'Transversal')


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
    ):
        self.data_path = data_path
        files = []
        n_slices = []
        for fn in Path(data_path).rglob('*.h5'):
            with h5py.File(fn, 'r') as f:
                location = f.attrs['location']
                orientation = f.attrs['orientation']
                if location in locations and orientation in orientations:
                    files.append(fn)
                    n_slices.append(f['b1'].shape[0])
        if not len(files):
            raise ValueError(f'No files found for locations {locations} and orientations {orientations} in {data_path}')
        self.files = files
        self.n_slices = n_slices
        self.size = (96, 128)

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
            b1 = torch.tensor(np.array(f['b1'][slice_idx]))
            mask = torch.tensor(np.array(f['mask'][slice_idx]))
            localizer = torch.tensor(np.array(f['localizer'][slice_idx]))
            location = locations.index(f.attrs['location'])
            orientation = orientations.index(f.attrs['orientation'])
            subject_idx = f.attrs['index']
        localizer = rearrange(torch.view_as_real(localizer), '... coils realimag -> (coils realimag) ...')
        b1 = rearrange(torch.view_as_real(b1), ' ... coils realimag -> (coils realimag) ...')
        mask = mask.unsqueeze(0).float()
        b1, mask, localizer = self.pad_or_random_crop(b1, mask, localizer)
        return {
            'localizer': localizer,
            'b1': b1,
            'mask': mask,
            'location': torch.tensor(location, dtype=torch.long),
            'orientation': torch.tensor(orientation, dtype=torch.long),
            'subject_idx': torch.tensor(subject_idx, dtype=torch.long),
            'slice_idx': torch.tensor(slice_idx, dtype=torch.long),
        }


class B1LocalizerModule(pl.LightningDataModule):
    """DataModule providing augmented inputs and targets."""

    def __init__(
        self,
        train_dir: str | Path,
        val_dir: str | Path,
        batch_size: int,
        num_workers: int,
        orientations: Sequence[Literal['Sagittal', 'Coronal', 'Transversal']] = orientations,
        locations: Sequence[Literal['Minnesota', 'Buch', 'Heidelberg']] = locations,
        augment: None | DataAugmentation = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.orientations = orientations
        self.locations = locations

    def setup(self, stage: str | None = None) -> None:
        if stage != 'fit':
            raise NotImplementedError(f'not implemeted yet: {stage}')
        self.train_dataset = B1LocalizerDS(self.train_dir, self.orientations, self.locations)
        self.val_dataset = B1LocalizerDS(self.val_dir, self.orientations, self.locations)

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

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int):
        if self.trainer.training and self.augment:
            localizer, b1, mask = self.augment(batch['localizer'], batch['b1'], batch['mask'])
            batch.update(localizer=localizer, b1=b1, mask=mask)
        return batch


class MaskedMSELoss(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        mask = mask.broadcast_to(pred.shape)
        loss = (mask * (pred - target)).square().sum()
        loss = loss / (mask.sum() + 1e-6)
        return loss


class B1Predictor(pl.LightningModule):
    """B1 prediction from localizer."""

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        n_features: tuple[int, ...],
        attention_depths: tuple[int, ...],
        append_rss: bool = True,
        n_rx: int = 32,
        n_tx: int = 8,
        embedding_dim: int = 64,
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
        )
        self.unet = torch.compile(self.unet, dynamic=False, fullgraph=True)
        self.location_embedding = torch.nn.Embedding(len(locations), embedding_dim)
        self.orientation_embedding = torch.nn.Embedding(len(orientations), embedding_dim)
        self.criterion = MaskedMSELoss()

    def forward(
        self,
        localizer: torch.Tensor,
        location: torch.Tensor,
        orientation: torch.Tensor,
    ):
        if self.hparams.append_rss:
            rss = localizer.square().abs().sum(dim=1).sqrt()
            localizer = torch.cat([localizer, rss[:, None]], dim=1)

        location = self.location_embedding(location)
        orientation = self.orientation_embedding(orientation)
        cond = location + orientation

        return self.unet(localizer, cond=cond)

    def training_step(self, batch: BatchDict, batch_idx: int):
        pred = self(batch['localizer'], batch['location'], batch['orientation'])
        loss = self.criterion(pred, batch['b1'], batch['mask'])
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: BatchDict, batch_idx: int):
        pred = self(batch['localizer'], batch['location'], batch['orientation'])
        loss = self.criterion(pred, batch['b1'], batch['mask'])
        self.log('val_loss', loss, sync_dist=True)

        def to_complex(x):
            x = rearrange(x, ' batch (coils realimag) ... -> batch coils ... realimag', realimag=2)
            x = torch.view_as_complex(x.contiguous())
            return x * batch['mask']

        gt_complex = to_complex(batch['b1'])
        pred_complex = to_complex(pred)
        ssim_magnitude = structural_similarity_index_measure(pred_complex.abs(), gt_complex.abs(), data_range=1.0)
        ssim_phase = structural_similarity_index_measure(
            pred_complex.angle(), gt_complex.angle(), data_range=(-np.pi, np.pi)
        )

        self.log('val_ssim_magnitude', ssim_magnitude, sync_dist=True)
        self.log('val_ssim_phase', ssim_phase, sync_dist=True)

        subject_idx = batch['subject_idx'][0].item()
        slice_idx = batch['slice_idx'][0].item()

        if subject_idx == 4 and slice_idx == 5:
            n_tx = self.hparams.n_tx
            fig, axes = plt.subplots(4, n_tx, figsize=(n_tx * 2, 8), constrained_layout=True)
            nan_mask = np.where(batch['mask'][0].cpu().numpy() < 0.5, np.nan, 1)
            gt_mag = gt_complex.abs()[0].cpu().numpy() * nan_mask
            pred_mag = pred_complex.abs()[0].cpu().numpy() * nan_mask
            gt_phase = gt_complex.angle()[0].cpu().numpy() * nan_mask
            pred_phase = pred_complex.angle()[0].cpu().numpy() * nan_mask

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

            location_str = locations[batch['location'][0].item()]
            orientation_str = orientations[batch['orientation'][0].item()]
            fig.suptitle(f'S:{subject_idx} Sl:{slice_idx} L:{location_str} O:{orientation_str}')
            self.logger.run[f'val_images/{location_str}/{orientation_str}'].log(fig)
            plt.close(fig)

        return loss

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched_g = OneCycleLR(
            opt_g,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.2,
            final_div_factor=20,
            div_factor=25,
        )
        return [opt_g], [{'scheduler': sched_g, 'interval': 'step'}]


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch._inductor.config.worker_start_method = 'fork'
    torch._inductor.config.compile_threads = 4
    torch._dynamo.config.capture_scalar_outputs = True
    # torch._functorch.config.activation_memory_budget = 0.9

    dm = B1LocalizerModule(
        train_dir='/echo/allgemein/projects/MRpro/B1/FFZHK/processed/train',
        val_dir='/echo/allgemein/projects/MRpro/B1/FFZHK/processed/val',
        batch_size=4,
        num_workers=4,
        augment=DataAugmentation(),
    )

    model = B1Predictor(
        lr=2e-4,
        weight_decay=1e-4,
        n_features=(64, 128, 192, 256),
        attention_depths=(-1, -2),
        append_rss=True,
    )

    logger = NeptuneLogger(
        project='ptb/b1p',
        log_model_checkpoints=False,
        dependencies='infer',
        source_files=[__file__],
        api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyOTdlYTM3NS0wMWU1LTRlMzMtYWU1Ny01MzMzN2ExNTcwMDcifQ==',
    )
    logger.log_model_summary(model=model, max_depth=-1)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        dirpath=f'checkpoints/b1/{logger.version}',
        filename=f'b1-{logger.version}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_last=True,
    )
    ddp_strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            checkpoint_callback,
        ],
        log_every_n_steps=2,
        check_val_every_n_epoch=4,
        precision='16-mixed',
        # strategy=ddp_strategy,
    )

    trainer.fit(model, datamodule=dm)
