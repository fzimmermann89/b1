# B1 Prediction

learned B1+ prediction from localizers.

## Installation

```bash
# Install the package in development mode
pip install -e .
```

## Usage

### Training Data Pre Processing

```bash
# Process raw data
python data_processing.py path
```

### Training

You need to have a neptune account and set the neptune API key as an environment variable.

```bash
# Basic training
python -m b1prediction.app fit

# or
b1 fit

# With custom parameters
python -m b1prediction.app fit \
    --model.lr 1e-3 \
    --data.batch_size 16 \
    --data.fold 0 \
    --data.n_folds 4 \
    --experiment "My Experiment" \
    --neptune_project "myproject/b1p"
```

## Hyperparameter Tuning

The project supports automated hyperparameter optimization using Optuna with distributed training on SLURM clusters.

### Running HPO

```bash

# Start tuning with a configuration file
b1 tune --config hpo.yaml
```

### HPO Configuration

Create a YAML configuration file (e.g., `hpo.yaml`) to define the hyperparameter search space, trainer settings, and SLURM configuration:

```yaml
# Example hpo.yaml configuration
trainer:
  max_epochs: 50

slurm_config:
  num_workers: 8
  gres: 'shard:1'

hpo_config:
  n_trials: 40
  storage: 'sqlite:///optuna.db'

search_space:
  - name: lr
    type: 'float'
    low: 1e-5
    high: 1e-3
    log: true

  - name: weight_decay
    type: 'float'
    low: 1e-6
    high: 1e-2
    log: true

  - name: n_features
    type: categorical
    choices:
      - [64, 128, 192, 192]
      - [64, 128, 256, 256]
      - [64, 96, 128, 192]

  - name: attention_depths
    type: categorical
    choices:
      - [-1, -2]
      - []
      - [-1]

  - name: embedding_dim
    type: categorical
    choices:
      - 64
      - 128

  - name: p_affine
    type: float
    low: 0.0
    high: 0.9
```

### Monitoring HPO Progress

```bash
# Monitor trials in the Optuna database
optuna trials --study-name "your_study_name" --storage "sqlite:///optuna.db"
```

## Relevant Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 5e-4 | Learning rate for the optimizer |
| `weight_decay` | float | 1e-4 | Weight decay for regularization |
| `n_features` | tuple[int] | (64, 128, 192, 192) | Number of features per scale in UNet |
| `attention_depths` | tuple[int] | (-1, -2) | Which scales to apply attention to |
| `append_rss` | bool | True | Whether to append RSS (root sum of squares) to input |
| `embedding_dim` | int | 128 | Dimension of location/orientation embeddings |
| `p_dropout_cond` | float | 0.2 | Dropout probability for conditioning MLP |
| `fold` | int | 0 | Current fold for cross-validation |
| `n_folds` | int | 4 | Total number of cross-validation folds |
| `orientations` | tuple[str] | ('Sagittal', 'Coronal', 'Transversal') | MRI scan orientations to include |
| `locations` | tuple[str] | ('Buch', 'Minnesota', 'Heidelberg') | Data collection locations to include |
| `p_affine` | float | 0.75 | Probability of applying affine augmentation |
| `affine_degrees` | float | 5.0 | Maximum rotation in degrees for augmentation |
| `affine_translate` | float | 0.02 | Maximum translation for augmentation |
| `affine_scale` | float | 0.02 | Maximum scale change for augmentation |
| `max_epochs` | int | 400 | Maximum number of training epochs |
| `precision` | str | '16-mixed' | Training precision (16-bit mixed precision) |
