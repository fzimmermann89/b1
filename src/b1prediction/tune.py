"""SLURM-based hyperparameter tuning logic for B1 prediction."""

import shlex
import subprocess
import sys
from typing import Any

import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

from b1prediction.data import B1LocalizerModule
from b1prediction.model import B1Predictor


def launch_slurm_jobs(cli: Any) -> None:
    """Create an Optuna study and submit worker jobs to SLURM."""
    print('--- Starting SLURM HPO Master ---')
    cfg = cli.config.tune

    optuna.create_study(
        storage=cfg.hpo.storage,
        study_name=cfg.hpo.study_name,
        direction=cfg.hpo.direction,
        load_if_exists=True,
    )
    print(f"Optuna study '{cfg.hpo.study_name}' at: {cfg.hpo.storage}")

    script_name, command_name, *original_args = sys.argv
    quoted_args = ' '.join(shlex.quote(arg) for arg in original_args)
    worker_command = f'{script_name} {command_name} --worker {quoted_args}'

    sbatch_command = f"""sbatch \\
--job-name={cfg.slurm.job_name} --partition={cfg.slurm.partition} --gres={cfg.slurm.gres} \\
--cpus-per-task={cfg.slurm.cpus_per_task} --mem={cfg.slurm.mem} --time={cfg.slurm.time} \\
--wrap="{worker_command}"
"""
    print('\n--- Submitting SLURM Jobs ---')
    for i in range(cfg.slurm.num_workers):
        print(f'Submitting worker {i + 1}/{cfg.slurm.num_workers}...')
        subprocess.run(sbatch_command, shell=True, check=True)

    print(f'\n{cfg.slurm.num_workers} workers submitted. Monitor with `squeue`.')
    print(f'To see best params: optuna studies --study-name "{cfg.hpo.study_name}" --storage "{cfg.hpo.storage}"')


def run_worker(cli: Any) -> None:
    """Run a single Optuna worker process."""
    cfg = cli.config.tune
    study = optuna.load_study(study_name=cfg.hpo.study_name, storage=cfg.hpo.storage)

    def objective(trial: optuna.trial.Trial) -> float:
        model_cfg = dict(cfg.model.init_args)

        for name, param in cfg.search_space.items():
            if param.type == 'float':
                model_cfg[name] = trial.suggest_float(name, param.low, param.high, log=param.log, step=param.step)
            elif param.type == 'int':
                model_cfg[name] = trial.suggest_int(name, param.low, param.high, step=param.step)
            elif param.type == 'categorical':
                model_cfg[name] = trial.suggest_categorical(name, param.choices)

        model_cfg['plot_validation_images'] = False  # Disable plotting for HPO
        model = B1Predictor(**model_cfg)
        datamodule = B1LocalizerModule(**cfg.data.init_args)

        logger = NeptuneLogger(
            project=cfg.neptune_project,
            name=f'{cfg.experiment}-trial-{trial.number}',
            tags=[cfg.experiment, 'hpo', f'trial_{trial.number}'],
        )
        logger.run['hpo/params'] = trial.params

        trainer = Trainer(
            **cfg.trainer,
            logger=logger,
            callbacks=[EarlyStopping(monitor=cfg.hpo.metric, patience=10, mode='min')],
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, datamodule=datamodule)
        return trainer.callback_metrics[cfg.hpo.metric].item()

    study.optimize(objective, n_trials=cfg.hpo.n_trials, n_jobs=1)
