from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
import h5py
from utils import add_white_noise, ispadding
import random
import mne
import os

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

        # FIX: Initialize raw_nmm based on Forward Matrix columns (the full source space)
        # This prevents the broadcasting error with the (500, 76) data.
        num_sources = self.fwd.shape[1]
        raw_nmm = np.zeros((500, num_sources))

        for kk in range(raw_lb.shape[0]):
            curr_lb = raw_lb[kk, ~ispadding(raw_lb[kk])]
            
            # current_nmm_data is (500, 76) as per your H5 file
            current_nmm_data = self.data[self.dataset_meta['nmm_idx'][index][kk]]

            # Process activity for the source patch
            ssig = current_nmm_data[:, [0]] # Reference signal from the patch
            ssig = ssig / (np.max(ssig) + 1e-9)
            ssig *= self.dataset_meta['scale_ratio'][index][kk][
                random.randint(0, self.num_scale_ratio - 1)
            ]

            weight_decay = self.dataset_meta['mag_change'][index][kk]
            weight_decay = weight_decay[~ispadding(weight_decay)]

            # FIX: Map the signal from current_nmm_data into the full brain space
            # We only use as many columns as exist in the H5 slice (76)
            num_to_fill = min(len(curr_lb), current_nmm_data.shape[1])
            raw_nmm[:, curr_lb[:num_to_fill]] += ssig * weight_decay[:num_to_fill]

        # MULTIPLICATION: (Electrodes, Sources) @ (Sources, Time)
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

        # FIX: Same shape fix as SpikeEEGBuild
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

            weight_decay = self.dataset_meta['mag_change'][index][kk]
            weight_decay = weight_decay[~ispadding(weight_decay)]

            # Map into full brain space
            num_to_fill = min(len(curr_lb), current_nmm_data.shape[1])
            raw_nmm[:, curr_lb[:num_to_fill]] += ssig * weight_decay[:num_to_fill]

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