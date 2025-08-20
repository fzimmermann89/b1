import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

MAX_B1 = 120


def process(data: np.ndarray, location: str, orientation: str) -> np.ndarray:
    """flip, rotate, resize etc."""
    is_localizer = data.ndim == 4 and data.shape[0] == 32
    is_b1 = data.ndim == 4 and data.shape[0] == 8
    if 'Min' in location and 'Trans' not in orientation:
        if data.shape[-1] > 64:
            data = data[..., ::2]
        data = np.rot90(data, k=-1, axes=(-1, -2))
    if 'Sag' in orientation and 'Min' not in location:
        data = np.flip(data, axis=-2)
    if 'Cor' in orientation:
        if not ('Heidel' in location and is_localizer):
            data = np.flip(data, axis=-2)
        else:
            data = np.flip(data, axis=-1)
    if 'Min' in location and is_b1:
        data = np.roll(data, shift=1, axis=0)
    if is_b1 or is_localizer:
        data = np.moveaxis(data, 0, -1)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process B1 localizer data.')
    parser.add_argument('data_path', type=Path, help='Path to the root data directory')
    args = parser.parse_args()
    data_path = args.data_path

    for locationpath in filter(lambda x: x.is_dir(), data_path.iterdir()):
        for orientationpath in filter(lambda x: x.is_dir(), locationpath.iterdir()):
            fn: Path
            for i, fn in enumerate(tqdm(sorted(list(orientationpath.glob('*.mat'))))):
                orientation = str(fn.parent.stem)
                location = str(fn.parent.parent.stem)

                with h5py.File(fn, 'r') as f:
                    localizer = process(
                        np.array(f['localizer']['real']) + 1j * np.array(f['localizer']['imag']), location, orientation
                    ).astype(np.complex64)
                    b1 = np.array(f['CombB1RFromAFI']['real'] + 1j * np.array(f['CombB1RFromAFI']['imag']))
                    mean_phase = np.angle(np.mean(b1[1:6], axis=0, keepdims=True))
                    b1 = b1 * np.exp(-1j * mean_phase)
                    b1 = process(b1, location, orientation).astype(np.complex64)
                    mask = process(np.array(f['mask_brain']).astype(bool), location, orientation)
                    localizer = np.nan_to_num(localizer)
                    localizer = localizer / np.abs(localizer).max()
                    bad_values = np.any(np.abs(b1) > 120, axis=-1)
                    b1[np.abs(b1) > MAX_B1] = MAX_B1 * np.exp(1j * np.angle(b1[np.abs(b1) > MAX_B1]))
                    b1 = np.nan_to_num(b1) / 90
                    mask = mask & ~bad_values
                    outpath = data_path / 'processed' / f'{location}_{orientation}_{i:02d}.h5'
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                    with h5py.File(outpath, 'w') as f:
                        f.create_dataset('localizer', data=localizer)
                        f.create_dataset('b1', data=b1)
                        f.create_dataset('mask', data=mask)
                        f.attrs['location'] = location
                        f.attrs['orientation'] = orientation
                        f.attrs['index'] = i
                        f.attrs['filename'] = str(fn.stem)
