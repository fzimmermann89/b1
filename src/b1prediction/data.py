"""Data loading and preprocessing for B1 prediction."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

import h5py
import kornia.augmentation
import lightning.pytorch
import numpy as np
import torch

from b1prediction.util import complex_to_real, real_to_complex


class BatchDict(TypedDict):
    """Batch dictionary for training data."""

    localizer: torch.Tensor
    b1: torch.Tensor
    mask: torch.Tensor
    location: torch.Tensor
    orientation: torch.Tensor
    subject_idx: torch.Tensor
    slice_idx: torch.Tensor


locations = ('Minnesota', 'Buch', 'Heidelberg')
orientations = ('Sagittal', 'Coronal', 'Transversal')


class B1LocalizerDS(torch.utils.data.Dataset):
    """Dataset for B1 localizer data."""

    def __init__(
        self,
        data_path: Path,
        orientations: Sequence[Literal['Sagittal', 'Coronal', 'Transversal']] = orientations,
        locations: Sequence[Literal['Minnesota', 'Buch', 'Heidelberg']] = locations,
        fold: int = 0,
        n_folds: int = 4,
        train: bool = True,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        data_path
            Path to the data directory
        orientations
            List of orientations to include
        locations
            List of locations to include
        fold
            Current fold for cross-validation
        n_folds
            Total number of folds
        train
            Whether this is training data
        """
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

    def pad_or_random_crop(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Pad or randomly crop tensors to target size.

        Parameters
        ----------
        *tensors
            Tensors to pad or crop

        Returns
        -------
            Padded or cropped tensors
        """
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

    def __len__(self):
        """Return the total number of slices."""
        return sum(self.n_slices)

    def __getitem__(self, idx: int) -> BatchDict:
        """Get a single sample."""
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


class B1LocalizerModule(lightning.pytorch.LightningDataModule):
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
        """Initialize the data module.

        Parameters
        ----------
        data_dir
            Directory containing the processed data
        batch_size
            Batch size for training
        num_workers
            Number of workers for data loading
        fold
            Current fold for cross-validation
        n_folds
            Total number of folds
        orientations
            List of orientations to include
        locations
            List of locations to include
        p_affine
            Probability of applying affine augmentation
        affine_degrees
            Maximum rotation in degrees
        affine_translate
            Maximum translation
        affine_scale
            Maximum scale change
        """
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
        """Set up the datasets.

        Parameters
        ----------
        stage
            Training stage
        """
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

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return training data loader.

        Returns
        -------
            Training data loader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return validation data loader.

        Returns
        -------
            Validation data loader
        """
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)

    def on_after_batch_transfer(self, batch: BatchDict, dataloader_idx: int) -> BatchDict:  # noqa: ARG002
        """Apply augmentation after batch transfer.

        Parameters
        ----------
        batch
            Batch data
        dataloader_idx
            DataLoader index

        Returns
        -------
        BatchDict
            Augmented batch data
        """
        if self.trainer.training:
            params = self.affine.forward_parameters(batch['mask'].shape)
            mask = self.affine(batch['mask'], params=params)
            localizer = real_to_complex(self.affine(complex_to_real(batch['localizer']), params=params))
            b1 = real_to_complex(self.affine(complex_to_real(batch['b1']), params=params))
            batch.update(localizer=localizer, b1=b1, mask=mask)
        return batch
