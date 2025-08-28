"""B1 prediction model and loss functions."""

import lightning.pytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from mrpro.nn.nets import UNet
from torchmetrics.functional import structural_similarity_index_measure

from b1prediction.data import BatchDict, locations, orientations
from b1prediction.util import complex_to_real, real_to_complex


class MaskedL1Loss(torch.nn.Module):
    """Masked L1 loss for complex data."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked L1 loss.

        Parameters
        ----------
        pred
            Predicted values
        target
            Target values
        mask
            Binary mask

        Returns
        -------
            Masked L1 loss
        """
        loss = (mask * (pred - target).abs()).sum()
        loss = loss / (mask.sum() + 1e-9)
        return loss


class MaskedMSELoss(torch.nn.Module):
    """Masked MSE loss for complex data."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked MSE loss.

        Parameters
        ----------
        pred
            Predicted values
        target
            Target values
        mask
            Binary mask

        Returns
        -------
            Masked MSE loss
        """
        loss = (mask * (pred - target).abs().square()).sum()
        loss = loss / (mask.sum() + 1e-9)
        return loss


class B1Predictor(lightning.pytorch.LightningModule):
    """B1 prediction from localizer.
    
    Uses a standard UNet from mrpro + a conditioning network
    """

    def __init__(
        self,
        lr: float = 5e-4,  # noqa: ARG002
        weight_decay: float = 1e-4,  # noqa: ARG002
        n_features: tuple[int, ...] = (64, 128, 192, 192),
        attention_depths: tuple[int, ...] = (-1, -2),
        append_rss: bool = True,
        n_rx: int = 32,
        n_tx: int = 8,
        embedding_dim: int = 128,
        loss: torch.nn.Module = MaskedMSELoss(),
        p_dropout_cond: float = 0.2,
        plot_validation_images: bool = True,  # noqa: ARG002
    ):
        """Initialize the B1 predictor model.

        Parameters
        ----------
        lr
            Learning rate
        weight_decay
            Weight decay for optimizer
        n_features
            Number of features per scale in UNet
        attention_depths
            Which scales to apply attention to
        append_rss
            Whether to append RSS to input
        n_rx
            Number of receive channels
        n_tx
            Number of transmit channels
        embedding_dim
            Dimension of location/orientation embeddings
        loss
            Loss function to use
        p_dropout_cond
            Dropout probability for conditioning MLP
        plot_validation_images
            Whether to plot validation images
        """
        super().__init__()
        self.save_hyperparameters(ignore=['loss'])

        # The actual network predicting the B1+
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
        
        # Additional information for the network about the location and orientation
        # This is used to create a "cond" for the unet based on location+orientation, that is 
        # used for Film conditioning inside the unet.
        self.location_embedding = torch.nn.Embedding(len(locations), embedding_dim // 2)
        self.orientation_embedding = torch.nn.Embedding(len(orientations), embedding_dim // 2)
        self.cond_mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.Dropout(p_dropout_cond),
            torch.nn.GELU(),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.Dropout(p_dropout_cond),
        )
        self.criterion = loss
        self.validation_plotted = set()

    def forward(
        self,
        localizer: torch.Tensor,
        location: torch.Tensor,
        orientation: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        localizer
            Localizer data
        location
            Location index
        orientation
            Orientation index

        Returns
        -------
            Predicted B1 values
        """
        localizer_real = complex_to_real(localizer)
        if self.hparams.append_rss:
            rss = localizer.abs().square().sum(dim=1).sqrt()
            localizer_real = torch.cat([localizer_real, rss[:, None]], dim=1)

        # create cond
        location_emb = self.location_embedding(location)
        orientation_emb = self.orientation_embedding(orientation)
        cond = torch.cat([location_emb, orientation_emb], dim=1)
        cond = self.cond_mlp(cond)

        prediction = self.unet(localizer_real, cond=cond)
        prediction = real_to_complex(prediction).to(torch.complex64)
        return prediction

    def training_step(self, batch: BatchDict, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Run training step.

        Parameters
        ----------
        batch
            Training batch
        batch_idx
            Batch index

        Returns
        -------
            Training loss
        """
        pred_complex = self(batch['localizer'], batch['location'], batch['orientation'])
        loss = self.criterion(pred_complex, batch['b1'], batch['mask'])
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: BatchDict, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Run validation step.

        Parameters
        ----------
        batch
            Validation batch
        batch_idx
            Batch index

        Returns
        -------
            Validation loss
        """
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

        if (
            self.hparams.plot_validation_images
            and slice_idx == 5
            and (orientation_idx, location_idx) not in self.validation_plotted # dont plot twice..
        ):
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

    def on_validation_epoch_end(self) -> None:
        """Clear validation plotted set at end of epoch."""
        self.validation_plotted.clear()

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """Configure optimizers and schedulers.

        Returns
        -------
        tuple[list[torch.optim.Optimizer], list[dict]]
            Tuple of (optimizers, schedulers)
        """
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
