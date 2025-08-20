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
