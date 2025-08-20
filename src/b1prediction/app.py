"""CLI application for B1 prediction training."""

import os
import shlex
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.trainer import Trainer

from b1prediction.data import B1LocalizerModule
from b1prediction.model import B1Predictor

if not sys.warnoptions:
    warnings.simplefilter('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'


class MyLightningCLI(LightningCLI):
    """Custom CLI."""

    def add_arguments_to_parser(self, parser):
        """Add custom arguments to the parser."""
        parser.add_argument('--experiment', type=str, default='B1 prediction', help='Name of the experiment.')
        parser.add_argument('--neptune_project', type=str, default='ptb/b1p', help='Neptune project name.')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/b1', help='path for checkpoints.')
        parser.set_defaults({'data': {'class_path': 'b1prediction.data.B1LocalizerModule'}})

    def before_instantiate_classes(self) -> None:
        """Set up logger and callbacks before instantiating classes."""
        config = getattr(self.config, self.subcommand)
        fold = config.data.init_args.fold
        experiment_name = config.experiment

        if self.subcommand != 'fit':
            return

        logger = NeptuneLogger(
            project=config.neptune_project,
            name=f'{experiment_name}-fold{fold}',
            tags=[experiment_name, f'fold_{fold}'],
            log_model_checkpoints=False,
            dependencies='infer',
            source_files=[__file__],
        )
        _ = logger.run
        config.trainer.logger = logger

        checkpoint_callback = ModelCheckpoint(
            monitor='validation/loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            dirpath=f'{config.checkpoint_dir}/{logger.version}',
            filename=f'{logger.version}-{{epoch:02d}}-{{val_loss:.4f}}',
        )

        config.trainer.callbacks = [
            LearningRateMonitor(logging_interval='step'),
            checkpoint_callback,
        ]

    def after_instantiate_classes(self) -> None:
        """Log model summary after the model and logger are instantiated."""
        if isinstance(self.trainer.logger, NeptuneLogger):
            self.trainer.logger.log_model_summary(model=self.model, max_depth=-1)
        config = getattr(self.config, self.subcommand)
        self.trainer.cli_config = config

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        return {
            'fit': {'model', 'train_dataloaders', 'val_dataloaders', 'datamodule'},
            'validate': {'model', 'dataloaders', 'datamodule'},
            'tune': {'model', 'datamodule'},
        }


@dataclass
class SlurmManager:
    """Manages SLURM job configuration and submission."""

    partition: str | None = None
    gres: str | None = 'gpu:1'
    cpus_per_task: int = 4
    mem: str = '16G'
    time: str = '24:00:00'
    num_workers: int = 4

    def submit(self, command: str, job_name: str) -> None:
        """Submit a job to SLURM."""
        sbatch_parts = ['sbatch']
        sbatch_parts.append(f'--job-name={job_name}')
        if self.partition is not None:
            sbatch_parts.append(f'--partition={self.partition}')
        if self.gres is not None:
            sbatch_parts.append(f'--gres={self.gres}')
        sbatch_parts.append(f'--cpus-per-task={self.cpus_per_task}')
        if self.mem is not None:
            sbatch_parts.append(f'--mem={self.mem}')
        if self.time is not None:
            sbatch_parts.append(f'--time={self.time}')
        sbatch_parts.append(f'--wrap="{command}"')
        sbatch_command = ' \\\n'.join(sbatch_parts)

        print(f'Submitting job: {job_name}')
        subprocess.run(sbatch_command, shell=True, check=True)


@dataclass
class HPOConfig:
    """Configuration for Optuna HPO with sensible defaults."""

    storage: str = 'sqlite:///optuna.db'
    metric: str = 'validation/loss'
    direction: str = 'minimize'
    n_trials: int = 100


@dataclass
class SearchSpaceItem:
    """Defines a single hyperparameter for Optuna to search."""

    type: Literal['float', 'int', 'categorical']
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[str | float | int] = field(default_factory=list)


class HPOTrainer(Trainer):
    """Custom trainer with hyperparameter tuning."""

    experiment: str
    neptune_project: str

    def tune(
        self,
        model: 'pl.LightningModule',
        datamodule: 'pl.LightningDataModule | None' = None,
        search_space: dict[str, Any] = {},
        hpo_config: HPOConfig = HPOConfig(),
        slurm_config: SlurmManager = SlurmManager(),
        worker: bool = False,
    ) -> None:
        """Run Optuna hyperparameter tuning on a SLURM cluster."""
        if worker:
            self._run_worker(model, datamodule, search_space, hpo_config)
        else:
            self._launch_slurm_jobs(slurm_config, hpo_config)

    def _launch_slurm_jobs(self, slurm_config: SlurmManager, hpo_config: HPOConfig) -> None:
        """Create an Optuna study and submit worker jobs to SLURM."""
        print('--- Starting SLURM HPO Master ---')
        study_name = f'{self.cli_config.experiment.lower().replace(" ", "_")}_HPO'
        optuna.create_study(
            storage=hpo_config.storage,
            study_name=study_name,
            direction=hpo_config.direction,
            load_if_exists=True,
        )
        print(f"Optuna study '{study_name}' at: {hpo_config.storage}")
        original_args = ' '.join(shlex.quote(arg) for arg in sys.argv[2:])
        worker_command = f'{sys.executable} {sys.argv[0]} tune --worker true {original_args}'

        print('\n--- Submitting SLURM Jobs ---')
        for _ in range(slurm_config.num_workers):
            slurm_config.submit(worker_command, job_name=study_name)

        print(f'\n{slurm_config.num_workers} workers submitted. Monitor with `squeue`.')
        print(f'To see best params: optuna studies --study-name "{study_name}" --storage "{hpo_config.storage}"')

    def _run_worker(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        search_space: dict[str, Any],
        hpo_config: HPOConfig,
    ) -> None:
        """Run a single Optuna worker process."""
        study = optuna.load_study(study_name=f'{self.cli_config.experiment}_HPO', storage=hpo_config.storage)

        def objective(trial: optuna.trial.Trial) -> float:
            model_cfg = dict(model.hparams)
            search_items = {k: SearchSpaceItem(**v) for k, v in search_space.items()}

            for name, param in search_items.items():
                if param.type == 'float':
                    model_cfg[name] = trial.suggest_float(name, param.low, param.high, log=param.log, step=param.step)
                elif param.type == 'int':
                    model_cfg[name] = trial.suggest_int(name, param.low, param.high, step=param.step)
                elif param.type == 'categorical':
                    model_cfg[name] = trial.suggest_categorical(name, param.choices)

            model_cfg['plot_validation_images'] = False
            del model_cfg['_instantiator']
            trial_model = B1Predictor(**model_cfg)

            logger = NeptuneLogger(
                project=self.cli_config.neptune_project,
                name=f'{self.cli_config.experiment}-trial-{trial.number}',
                tags=[self.cli_config.experiment, 'hpo', f'trial_{trial.number}'],
            )
            logger.run['hpo/params'] = trial.params
            trainer_cfg = dict(self.cli_config.trainer)
            trainer_cfg['logger'] = logger
            trainer_cfg['enable_checkpointing'] = False
            trainer_cfg['enable_progress_bar'] = False
            trainer_cfg['callbacks'] = [
                pl.callbacks.EarlyStopping(monitor=hpo_config.metric, patience=10, mode=hpo_config.direction[:3])
            ]
            trial_trainer = Trainer(**trainer_cfg)
            trial_trainer.fit(trial_model, datamodule=datamodule)
            return trial_trainer.callback_metrics[hpo_config.metric].item()

        study.optimize(objective, n_trials=hpo_config.n_trials, n_jobs=1)


def main() -> None:
    """Run program."""
    torch.set_float32_matmul_precision('high')
    torch._inductor.config.worker_start_method = 'fork'
    torch._inductor.config.compile_threads = 4
    torch._dynamo.config.capture_scalar_outputs = True

    MyLightningCLI(
        B1Predictor,
        B1LocalizerModule,
        save_config_callback=None,
        subclass_mode_data=True,
        seed_everything_default=123,
        trainer_class=HPOTrainer,
        trainer_defaults={
            'max_epochs': 400,
            'accelerator': 'gpu',
            'devices': 1,
            'precision': '32-true',
            'log_every_n_steps': 2,
            'check_val_every_n_epoch': 4,
        },
    )


if __name__ == '__main__':
    main()
