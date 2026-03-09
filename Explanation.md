# Training data 
- The .h5 file actually contains the EEG signals and .mat files contain information about those EEG signals 
- We created 640 simulations of EEG recordings which corresponds to 640 rows in .mat file 
- In one simulation 32 brain regions were active and 


- The shape of .h5 file is 500,76 means 500 timepoints and 76 active regions 

### Role of .mat file 
- The .mat file contains metadata for the simulations. It has 5 variables and each variable contains 640 entries since 640 simulations were generated.
##### i) selected_region : (640×1) Stores the seed brain region used for each simulation. This seed is used by the simulation algorithm to generate the active source patch (32 sources).
##### ii) nmm_idx (640×1) : Acts as an index pointer that indicates where the corresponding signal for that simulation is stored in
the .h5 dataset.
##### ii) sensor_snr (640×1) : Stores the signal-to-noise ratio used when adding noise to the EEG signals for each simulation.
##### iv) mag_change (640×1) : Represents the amplitude scaling factor applied to the simulated brain signals.
##### v) scale_ratio (640×1×2) : Stores the normalization parameters [min, max] used to scale the signals before they were used by the model.

# Training Part 
#### 1) Loaders.py 
- This file converts our .mat and .h5 simulation files to training samples for the neural network 
- When training starts pytorch first creates a dataset here : dataset = SpikeEEGBuild(data_root, fwd)
- class SpikeEEGBuild(Dataset) This class builds training samples 
- def __init__(self, data_root, fwd, transform=None, args_params=None): This function runs once at the beginning which does the following tasks

##### i) Load the .mat file
- self.dataset_meta = loadmat(self.file_path)
- This loads metadata of the .mat file containing:
  - selected_region
  - nmm_idx
  - sensor_snr
  - mag_change
  - scale_ratio

##### ii) Determine the dataset size
- self.dataset_len = self.dataset_meta['selected_region'].shape[0]
- This gives the number of simulations in the dataset
- In our case it becomes **640**, meaning there are **640 simulation samples**

##### iii) Store the forward matrix
- self.fwd = fwd
- This forward matrix converts **brain source activity → EEG signals**
- Shape of forward matrix is **76 × 994**
  - 76 EEG sensors
  - 994 brain regions

##### iv) Store scale ratio count
- self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]
- This tells how many possible scaling values exist for the signal amplitude

### 2) __getitem__(index)  (Most important function)
- This function runs every time PyTorch asks for a training sample
Example: dataset[10]-> This builds **one training example**

##### i) Load the .h5 file (only first time)
if self.data is None:
self.data = h5py.File(base + '_nmm.h5', 'r')['data']

- This loads the H5 dataset containing EEG signals
- Shape of dataset: (1003, 500, 76)
Meaning:
- 1003 EEG samples
- 500 time points
- 76 EEG channels

Each sample therefore has shape:(500 × 76)

##### ii) Get active brain regions for this simulation

raw_lb = self.dataset_meta['selected_region'][index]
- This tells which brain regions were active in this simulation

Example: [120, 145, 200 ...]

##### iii) Remove padding values
lb = raw_lb[~ispadding(raw_lb)]

- Some arrays contain padding values used only for storage
- These values are removed so only real active regions remain

Example: Before cleaning: [120,145,200,-1,-1] and After cleaning: [120,145,200]

##### iv) Create empty brain activity matrix
num_sources = self.fwd.shape[1]
raw_nmm = np.zeros((500, num_sources))

Example:(500 × 994)

Meaning:
- 500 time points
- 994 brain regions

Initially all regions have **zero activity**

##### v) Loop through each source patch

for kk in range(raw_lb.shape[0]):
- Each simulation can contain multiple active source patches
- The loader processes them one by one

##### vi) Get region indices for the patch
curr_lb = raw_lb[kk]

Example:[120,121,122,123] These neighboring regions form one **source patch**

##### vii) Get corresponding EEG signal from H5
current_nmm_data = self.data[self.dataset_meta['nmm_idx'][index][kk]]
- nmm_idx tells which signal to fetch from the H5 dataset

Example: nmm_idx = 25

So the loader fetches: self.data[25] of Shape: (500 × 76)

##### viii) Extract spike waveform
ssig = current_nmm_data[:, [0]]

- The first column is used as the reference spike waveform

Shape becomes:(500 × 1)

##### ix) Normalize signal
ssig = ssig / (np.max(ssig) + 1e-9)
- This scales the signal so the amplitude stays within a stable range

##### x) Apply random amplitude scaling
ssig *= scale_ratio[random_index]

- The spike amplitude is randomly scaled using scale_ratio values
- This introduces variability in signal strength

Example scale ratios: 0.8, 1.0, 1.2

##### xi) Apply spatial decay
weight_decay = mag_change
Example:[1.0, 0.7, 0.4, 0.2]

Meaning:
- Center brain region → strongest signal
- Neighboring regions → weaker signal

##### xii) Insert signal into full brain activity
raw_nmm[:, curr_lb] += ssig * weight_decay
- The spike signal is placed into the full brain activity matrix
- Only the active regions receive signals

Now raw_nmm represents: Brain activity across **994 regions over time**

##### xiii) Generate EEG signals using forward model
eeg = self.fwd @ raw_nmm.T

Matrix multiplication: (Channels × Sources) × (Sources × Time)
Example: (76 × 994) × (994 × 500) which gives Result: (76 × 500)
This produces the simulated **EEG signals**

##### xiv) Add noise to EEG
noisy_eeg = add_white_noise(eeg, csnr)
Noise level is taken from: sensor_snr

Example: 5 dB, 10 dB

##### xv) Normalize EEG signal
Three steps are applied:
- subtract channel mean
- subtract time mean
- divide by maximum amplitude

This stabilizes the signal for neural network training

##### xvi) Create ground truth source activity
target = np.zeros_like(raw_nmm)
target[:, lb] = raw_nmm[:, lb]

- Only active brain regions keep their signals
- All other regions remain zero

##### xvii) Normalize target
target /= (np.max(target) + 1e-9)
- Keeps target values within stable range

##### xviii) Create final training sample
sample = {
'data': noisy_eeg,
'nmm': target,
'label': raw_lb,
'snr': csnr
}

Meaning:
- data → EEG input signal
- nmm → true brain source activity
- label → indices of active regions
- snr → noise level

##### xix) Return sample
return sample
The neural network receives: Input → EEG signal Target → brain source activity

### 3) __len__()

def __len__(self): return self.dataset_len

- This tells PyTorch how many samples exist in the dataset
- In this dataset: 640 samples

### 4) Other classes in loaders.py
#### SpikeEEGLoad
- Used when data is already saved as individual .mat samples
- It simply loads data instead of constructing it
#### SpikeEEGBuildEval
- Similar to SpikeEEGBuild but used for evaluation/testing
- Supports additional features like:
  - realistic noise
  - frequency filtering