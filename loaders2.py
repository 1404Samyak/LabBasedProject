from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
import h5py
from utils import add_white_noise, ispadding
import random
import os

# ============================
# Load region-to-dipole neighbor mapping
# ============================
mapping_data = loadmat('/content/anatomy/fs_cortex_20k_region_mapping.mat')
nbs = mapping_data['nbs']  # neighbors for each region

# ============================
# SpikeEEGBuild
# ============================


class SpikeEEGBuild(Dataset):
    def __init__(self, data_root, fwd, transform=None, args_params=None):
        if args_params is None:
            args_params = {}

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform

        self.dataset_meta = loadmat(self.file_path)
        self.data = None

        self.dataset_len = args_params.get(
            'dataset_len',
            self.dataset_meta['selected_region'].shape[0]
        )

        self.num_scale_ratio = args_params.get(
            'num_scale_ratio',
            self.dataset_meta['scale_ratio'].shape[2]
        )

    def __getitem__(self, index):
        if self.data is None:
            base = os.path.splitext(self.file_path)[0]
            self.data = h5py.File(base + '_nmm.h5', 'r')['data']

        raw_lb = self.dataset_meta['selected_region'][index].astype(int)
        lb = raw_lb[~ispadding(raw_lb)]

        num_sources = self.fwd.shape[1]
        raw_nmm = np.zeros((500, num_sources))  # full brain space

        # print("The raw_lb is ",raw_lb)

        for kk in range(raw_lb.shape[0]):
            curr_lb = raw_lb[kk, ~ispadding(raw_lb[kk])]
            current_nmm_data = self.data[self.dataset_meta['nmm_idx'][index][kk]]

            #print("The curr_lb is ",curr_lb)
            # Normalize spike signal
            ssig = current_nmm_data[:, [0]]
            ssig = ssig / (np.max(ssig) + 1e-9)
            ssig *= self.dataset_meta['scale_ratio'][index][kk][
                random.randint(0, self.num_scale_ratio - 1)
            ]

            # ---------- DIPLOLE NEIGHBOR MAPPING WITH GAUSSIAN DECAY ----------
            region_id = curr_lb[0]  # center dipole
            neighbors = nbs[0][region_id].flatten()

            # Gaussian decay based on neighbor index (first = center, farther = less weight)
            sigma = 1.0
            neighbor_weights = np.exp(-np.arange(len(neighbors))**2 / (2*sigma**2))

            # ---------- DIPLOLE NEIGHBOR MAPPING WITH GAUSSIAN DECAY ----------
        region_id = curr_lb[0]  # center region
        neighbors = nbs[0][region_id].flatten()
        sigma = 1.0
        neighbor_weights = np.exp(-np.arange(len(neighbors))**2 / (2 * sigma**2))
        # Ensure valid indices (important safety)
        valid_len = min(len(neighbors), len(neighbor_weights))
        for i in range(valid_len):
            dipole_idx = neighbors[i]
            # Safety check (avoid index overflow)
            if dipole_idx >= raw_nmm.shape[1]:
                continue
            raw_nmm[:, dipole_idx] += ssig.flatten() * neighbor_weights[i]

            # print(region_id)
            # print(neighbors)
            # print(neighbor_weights)

        # Generate EEG
        eeg = self.fwd @ raw_nmm.T
        csnr = self.dataset_meta['sensor_snr'][index]

        noisy_eeg = add_white_noise(eeg, csnr).T
        noisy_eeg -= noisy_eeg.mean(axis=0, keepdims=True)
        noisy_eeg -= noisy_eeg.mean(axis=1, keepdims=True)
        noisy_eeg /= (np.max(np.abs(noisy_eeg)) + 1e-9)

        target = np.zeros_like(raw_nmm)
        target[:, lb] = raw_nmm[:, lb]
        target /= (np.max(target) + 1e-9)

        sample = {
            'data': noisy_eeg.astype(np.float32),
            'nmm': target.astype(np.float32),
            'label': raw_lb,
            'snr': csnr
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.dataset_len


# ============================
# SpikeEEGLoad
# ============================
class SpikeEEGLoad(Dataset):
    def __init__(self, data_root, fwd, transform=None, args_params=None):
        if args_params is None:
            args_params = {}
        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform
        self.dataset_len = args_params.get('dataset_len', 0)

    def __getitem__(self, index):
        raw_data = loadmat(f'{self.file_path}/data{index}.mat')
        sample = {
            'data': raw_data['data'].astype(np.float32),
            'nmm': raw_data['nmm'].astype(np.float32),
            'label': raw_data['label'],
            'snr': raw_data['csnr']
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.dataset_len


# ============================
# SpikeEEGBuildEval
# ============================
class SpikeEEGBuildEval(Dataset):
    def __init__(self, data_root, fwd, transform=None, args_params=None):
        if args_params is None:
            args_params = {}
        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform
        self.dataset_meta = loadmat(self.file_path)
        self.data = None
        self.eval_params = {}

        self.dataset_len = args_params.get(
            'dataset_len',
            self.dataset_meta['selected_region'].shape[0]
        )
        self.num_scale_ratio = args_params.get(
            'num_scale_ratio',
            self.dataset_meta['scale_ratio'].shape[2]
        )

        if args_params.get('snr_rsn_ratio', None):
            self.eval_params['rsn'] = loadmat('anatomy/realistic_noise.mat')
            self.eval_params['snr_rsn_ratio'] = args_params['snr_rsn_ratio']

        if args_params.get('lfreq', 0) > 0 and args_params.get('hfreq', 0) > 0:
            self.eval_params['lfreq'] = args_params['lfreq']
            self.eval_params['hfreq'] = args_params['hfreq']

    def __getitem__(self, index):
        if self.data is None:
            base = os.path.splitext(self.file_path)[0]
            self.data = h5py.File(base + '_nmm.h5', 'r')['data']

        raw_lb = self.dataset_meta['selected_region'][index].astype(int)
        lb = raw_lb[~ispadding(raw_lb)]

        num_sources = self.fwd.shape[1]
        raw_nmm = np.zeros((500, num_sources))

        for kk in range(raw_lb.shape[0]):
            curr_lb = raw_lb[kk, ~ispadding(raw_lb[kk])]
            current_nmm_data = self.data[self.dataset_meta['nmm_idx'][index][kk]]

            ssig = current_nmm_data[:, [0]]
            ssig = ssig / (np.max(ssig) + 1e-9)
            
            ssig *= self.dataset_meta['scale_ratio'][index][kk][
                random.randint(0, self.num_scale_ratio - 1)
            ]
            
            # DIPLOLE NEIGHBOR GAUSSIAN DECAY
            region_id = curr_lb[0]
            neighbors = nbs[0][region_id].flatten()
            sigma = 1.0
            neighbor_weights = np.exp(-np.arange(len(neighbors))**2 / (2*sigma**2))
            num_to_fill = min(len(curr_lb), current_nmm_data.shape[1], len(neighbor_weights))
            raw_nmm[:, curr_lb[:num_to_fill]] += ssig * neighbor_weights[:num_to_fill]
            
            print(region_id)
            print(neighbor_weights)
            print(neighbors)

        eeg = self.fwd @ raw_nmm.T
        csnr = self.dataset_meta['sensor_snr'][index]

        if 'rsn' in self.eval_params:
            noisy_eeg = add_white_noise(
                eeg, csnr,
                {
                    'ratio': self.eval_params['snr_rsn_ratio'],
                    'rndata': self.eval_params['rsn']['data'],
                    'rnpower': self.eval_params['rsn']['npower']
                }
            ).T
        else:
            noisy_eeg = add_white_noise(eeg, csnr).T

        noisy_eeg -= noisy_eeg.mean(axis=0, keepdims=True)
        noisy_eeg -= noisy_eeg.mean(axis=1, keepdims=True)
        noisy_eeg /= (np.max(np.abs(noisy_eeg)) + 1e-9)

        target = np.zeros_like(raw_nmm)
        target[:, lb] = raw_nmm[:, lb]
        target /= (np.max(target) + 1e-9)

        sample = {
            'data': noisy_eeg.astype(np.float32),
            'nmm': target.astype(np.float32),
            'label': raw_lb,
            'snr': csnr
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.dataset_len