"""CLI application for B1 prediction training."""

import ast
import dataclasses
import os
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Literal

import jsonargparse
import lightning.pytorch
import lightning.pytorch.callbacks
import lightning.pytorch.cli
import lightning.pytorch.loggers
import optuna
import torch
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from b1prediction.data import B1LocalizerModule
from b1prediction.model import B1Predictor

if not sys.warnoptions:
    warnings.simplefilter('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'


class MyLightningCLI(lightning.pytorch.cli.LightningCLI):
    """Custom CLI."""

    def add_arguments_to_parser(self, parser: jsonargparse.ArgumentParser) -> None:
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

        logger = lightning.pytorch.loggers.NeptuneLogger(
            project=config.neptune_project,
            name=f'{experiment_name}-fold{fold}',
            tags=[experiment_name, f'fold_{fold}'],
            log_model_checkpoints=False,
            dependencies='infer',
            source_files=[str(o) for o in Path(__file__).parent.rglob('*.py')],
        )
        _ = logger.run
        config.trainer.logger = logger

        checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
            monitor='validation/loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            dirpath=f'{config.checkpoint_dir}/{logger.version}',
            filename=f'{logger.version}-{{epoch:02d}}-{{val_loss:.4f}}',
        )

        config.trainer.callbacks = [
            lightning.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
            checkpoint_callback,
        ]

    def after_instantiate_classes(self) -> None:
        """Log model summary after the model and logger are instantiated."""
        if isinstance(self.trainer.logger, lightning.pytorch.loggers.NeptuneLogger):
            self.trainer.logger.log_model_summary(model=self.model, max_depth=-1)
        config = getattr(self.config, self.subcommand)
        self.trainer.cli_config = config

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Define trainer functions to use as subcommands of the CLI."""
        return {
            'fit': {'model', 'train_dataloaders', 'val_dataloaders', 'datamodule'},
            'validate': {'model', 'dataloaders', 'datamodule'},
            'tune': {'model', 'datamodule'},
        }


@dataclasses.dataclass
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


@dataclasses.dataclass
class HPOConfig:
    """Configuration for Optuna HPO with sensible defaults."""

    storage: str = 'sqlite:///optuna.db'
    metric: str = 'validation/loss'
    direction: str = 'minimize'
    n_trials: int = 100
    sampler: optuna.samplers.BaseSampler = dataclasses.field(default_factory=optuna.samplers.TPESampler)
    pruner: optuna.pruners.BasePruner = dataclasses.field(
        default_factory=lambda: optuna.pruners.MedianPruner(n_warmup_steps=1000)
    )


@dataclasses.dataclass
class SearchSpaceItem:
    """Defines a single hyperparameter for Optuna to search."""

    name: str
    type: Literal['float', 'int', 'categorical', 'categorical_ast']
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[int | float | str] = dataclasses.field(default_factory=list)


class HPOTrainer(lightning.pytorch.Trainer):
    """Custom trainer with hyperparameter tuning."""

    experiment: str
    neptune_project: str

    def tune(
        self,
        model: lightning.pytorch.LightningModule,
        datamodule: lightning.pytorch.LightningDataModule | None = None,
        search_space: tuple[SearchSpaceItem, ...] = (),
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
        study_name = self.cli_config.experiment.lower().replace(' ', '_')
        optuna.create_study(
            storage=hpo_config.storage,
            study_name=study_name,
            direction=hpo_config.direction,
            load_if_exists=True,
            sampler=hpo_config.sampler,
            pruner=hpo_config.pruner,
        )
        print(f"Optuna study '{study_name}' at: {hpo_config.storage}")
        original_args = ' '.join(shlex.quote(arg) for arg in sys.argv[2:])
        worker_command = f'{sys.executable} {sys.argv[0]} tune --worker true {original_args}'

        print('\n--- Submitting SLURM Jobs ---')
        for _ in range(slurm_config.num_workers):
            slurm_config.submit(worker_command, job_name=study_name)

        print(f'\n{slurm_config.num_workers} workers submitted. Monitor with `squeue`.')
        print(f'To see trials: optuna trials --study-name "{study_name}" --storage "{hpo_config.storage}"')

    def _run_worker(
        self,
        model: lightning.pytorch.LightningModule,
        datamodule: lightning.pytorch.LightningDataModule,
        search_space: list[SearchSpaceItem],
        hpo_config: HPOConfig,
    ) -> None:
        """Run a single Optuna worker process."""
        study_name = self.cli_config.experiment.lower().replace(' ', '_')
        study = optuna.load_study(study_name=study_name, storage=hpo_config.storage)

        def objective(trial: optuna.trial.Trial) -> float:
            model_cfg = dict(model.hparams)
            data_cfg = dict(datamodule.hparams)

            for p in search_space:
                if p.name in model_cfg:
                    cfg = model_cfg
                elif p.name in data_cfg:
                    cfg = data_cfg
                else:
                    raise ValueError(f'Parameter {p.name} not found in model or datamodule configuration.')

                if p.type == 'float':
                    cfg[p.name] = trial.suggest_float(p.name, p.low, p.high, log=p.log, step=p.step)
                elif p.type == 'int':
                    cfg[p.name] = trial.suggest_int(p.name, p.low, p.high, step=p.step)
                elif p.type == 'categorical':
                    cfg[p.name] = trial.suggest_categorical(p.name, p.choices)
                elif p.type == 'categorical_ast':
                    cfg[p.name] = ast.literal_eval(trial.suggest_categorical(p.name, p.choices))

            model_cfg['plot_validation_images'] = False
            model_cfg = {k: v for k, v in model_cfg.items() if not k.startswith('_')}
            data_cfg = {k: v for k, v in data_cfg.items() if not k.startswith('_')}
            trial_model = B1Predictor(**model_cfg)
            trial_datamodule = B1LocalizerModule(**data_cfg)

            logger = lightning.pytorch.loggers.NeptuneLogger(
                project=self.cli_config.neptune_project,
                name=f'{self.cli_config.experiment}-trial-{trial.number}',
                tags=[self.cli_config.experiment, 'hpo', f'trial_{trial.number}'],
                source_files=[str(o) for o in Path(__file__).parent.rglob('*.py')],
            )
            logger.run['hpo/params'] = trial.params
            trainer_cfg = dict(self.cli_config.trainer)
            trainer_cfg['logger'] = logger
            trainer_cfg['enable_checkpointing'] = False
            trainer_cfg['enable_progress_bar'] = False
            trainer_cfg['callbacks'] = [
                lightning.pytorch.callbacks.EarlyStopping(
                    monitor=hpo_config.metric, patience=10, mode=hpo_config.direction[:3]
                ),
                PyTorchLightningPruningCallback(trial, monitor=hpo_config.metric),
            ]
            trial_trainer = lightning.pytorch.Trainer(**trainer_cfg)
            trial_trainer.fit(trial_model, datamodule=trial_datamodule)
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
            'precision': '16-mixed',
            'log_every_n_steps': 2,
            'check_val_every_n_epoch': 4,
        },
    )


if __name__ == '__main__':
    main()
