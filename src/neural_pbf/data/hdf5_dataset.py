import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5ThermalDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for offline HDF5 thermal surrogate datasets.

    This dataset avoids loading the massive arrays into RAM by keeping an open
    handle to the HDF5 file and streaming minibatches dynamically. It handles
    converting the compressed fp16 / uint8 snapshots back into fp32 PyTorch
    tensors ready for the neural network.
    """

    def __init__(self, h5_paths: str | list[str]):
        if isinstance(h5_paths, str):
            self.h5_path = h5_paths
            h5_paths = [h5_paths]
        else:
            self.h5_path = h5_paths[0] if h5_paths else ""

        self.h5_paths = h5_paths
        self._keys = []  # list of (path, key)

        for path in self.h5_paths:
            with h5py.File(path, "r") as f:
                if "samples" in f:
                    keys = list(f["samples"].keys())
                    self._keys.extend([(path, k) for k in keys])

        self.length = len(self._keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        path, sample_key = self._keys[idx]
        with h5py.File(path, "r") as f:
            group = f["samples"][sample_key]

            # Read arrays and cast back to float32
            T_in = torch.from_numpy(group["T_in"][:].astype(np.float32))
            Q = torch.from_numpy(group["Q"][:].astype(np.float32))
            T_target = torch.from_numpy(group["T_target"][:].astype(np.float32))
            T_lf = torch.from_numpy(group["T_lf"][:].astype(np.float32))
            mask = torch.from_numpy(group["mask"][:].astype(np.uint8))

            scalars = torch.from_numpy(group["scalars"][:].astype(np.float32))

            # Process parameters can optionally be returned or retrieved via attrs
            # For now, we return the core arrays needed by the surrogate

            return {
                "T_in": T_in,
                "Q": Q,
                "T_target": T_target,
                "T_lf": T_lf,
                "mask": mask,
                "t": scalars[0],
                "laser_x": scalars[1],
                "laser_y": scalars[2],
            }
