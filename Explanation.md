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
- This loads metadata of the .mat file containing following variables:
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
Example : [1.0, 0.7, 0.4, 0.2]

Meaning:
- Center brain region → strongest signal
- Neighboring regions → weaker signal

##### xii) Insert signal into full brain activity
raw_nmm[:, curr_lb] += ssig * weight_decay
- The spike signal is placed into the full brain activity matrix
- Only the active regions receive signals

Now raw_nmm represents: Brain activity across **994 regions over time**

##### xiii) Generate EEG signals using forward model
eeg = self.fwd @ raw_nmm.T (raw_nmm matrix after being transposed)

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

### 4) Other classes in loaders.py
#### SpikeEEGLoad
- Used when data is already saved as individual .mat samples
- It simply loads data instead of constructing it
#### SpikeEEGBuildEval
- Similar to SpikeEEGBuild but used for evaluation/testing
- Supports additional features like:
  - realistic noise
  - frequency filtering

#### Network.py
This file defines the neural network architecture used to learn the mapping between EEG signals (from sensors) and brain source activity (in brain regions).

The network is divided into two main stages:
1. Spatial Filtering → learns relationships between EEG sensors(76 eeg sensors)
2. Temporal Filtering → learns patterns over time

The main model that combines both parts is called TemporalInverseNet.(spatial filtering+temporal filtering)
1) MLPSpatialFilter (Spatial Processing)
"class MLPSpatialFilter(nn.Module)"

This class performs spatial filtering of EEG signals across sensors(76 sensors) using fully connected layers (MLP).
EEG signals come from multiple sensors, and this module learns how sensors interact with each other.Basically tries to learn how eeg signals coming from different sensors interact with each other 

Example flow:
EEG Sensors
↓
Learn sensor relationships
↓
Create spatial features

i) Initialization
def __init__(self, num_sensor, num_hidden, activation): This function runs once when the model is created.

Parameters:
- num_sensor → number of EEG sensors (example: 76)
- num_hidden → hidden feature size (example: 500)
- activation → activation function (ReLU, Tanh etc.)

Layers created inside the model:
fc11 : Linear(num_sensor → num_sensor)
fc12 : Linear(num_sensor → num_sensor)

fc21 : Linear(num_sensor → num_hidden)
fc22 : Linear(num_hidden → num_hidden)

fc23 : Linear(num_sensor → num_hidden)

value : Linear(num_hidden → num_hidden)

Purpose of these layers:
EEG sensors
↓
Transform sensor signals
↓
Extract spatial features

Activation function is created dynamically: self.activation = nn.__dict__[activation]()
Meaning the model can use activation functions like: ReLU,Tanh or Sigmoid

ii) Forward Pass
def forward(self, x): This defines how EEG data flows through the spatial network.

Step 1: First transformation with residual connection
x = activation(fc12(activation(fc11(x))) + x)

Flow:
Input EEG
↓
fc11
↓
activation
↓
fc12
↓
add original input
↓
activation

The + x part is called a residual connection, which helps:
- better training
- stable gradients

Step 2: Hidden feature transformation
x = activation(fc22(activation(fc21(x))) + fc23(x))

Flow:
Sensor features
↓
Transform into hidden representation

Step 3: Final spatial output
out['value'] = value(x)
out['value_activation'] = activation(out['value'])

Two outputs are stored:
value
value_activation

Usually the network uses value_activation.

2) TemporalFilter (Temporal Processing)
"class TemporalFilter(nn.Module)" : This module learns temporal patterns in EEG signals using LSTM (Recurrent Neural Network).
EEG signals are time series, so temporal modeling is important ,this learns the patterns in EEG signals across time

Example flow:
EEG signals across time
↓
LSTM
↓
Temporal brain activity patterns


i) Initialization
def __init__(self, input_size, num_source, num_layer, activation):

Parameters:
- input_size → spatial feature size (example: 500)
- num_source → number of brain regions (example: 994)
- num_layer → number of LSTM layers
- activation → activation function

LSTM creation:
self.rnns.append(
    nn.LSTM(input_size, num_source,
            batch_first=True,
            num_layers=num_layer)
)

Meaning:
Input : spatial features
Output : predicted brain source signals

ii) Forward Pass
def forward(self, x):

Step 1: Pass input through LSTM
x, _ = l(x)

The LSTM processes the temporal sequence.

Output is stored as: out['rnn'] = x

Conceptual output shape : Batch × Time × BrainSources

Example:
Batch
↓
500 time points
↓
994 brain regions

3) TemporalInverseNet (Main Network)
"class TemporalInverseNet(nn.Module)"
This is the main model that combines spatial and temporal modules.

It solves the EEG inverse problem:
EEG signals → Brain source activity

i) Initialization

def __init__(...)

Important parameters:
- num_sensor → number of EEG sensors
- num_source → number of brain regions
- rnn_layer → number of LSTM layers
- temporal_input_size → hidden feature size

Two modules are created:
Spatial Module
↓
MLPSpatialFilter

Temporal Module
↓
TemporalFilter

Code:self.spatial = spat

### utils.py
- This file contains utility/helper functions used during training, evaluation, and signal processing in the EEG source localization pipeline.
These functions are not neural network layers. Instead, they help with:

- Handling padding values in labels
- Identifying predicted brain regions
- Adding noise to simulated EEG signals
- Mapping region-level predictions to vertex-level brain activity

These utilities support both data preprocessing and evaluation of neural network outputs.

#### 1) ispadding (Detect Padding Values)
"def ispadding(x)"
This function identifies padding values in arrays that store brain region indices.
In many datasets, arrays are stored with a fixed size, even if the actual number of elements is smaller.
To fill the unused positions, a special padding number is used.

In this code the padding value is: 15213
So whenever the value 15213 appears, it means:
- This is not a real brain region
- This is just a placeholder

How it works
Step 1: Convert the input to integer
x = x.astype(np.int32, copy=False)
This ensures the values are treated as integers.

Step 2: Compare values with the padding number
return np.abs(x - 15213) < 1e-6
If a value is very close to 15213, it is marked as padding.


Example if the Input label array: [120, 145, 200, 15213, 15213] then the Output mask: [False, False, False, True, True]
Meaning:
- First three values → real brain regions
- Last two values → padding values

Purpose : This function helps remove padding values when extracting actual active brain regions.

Example usage:
lb = raw_lb[~ispadding(raw_lb)]
This keeps only real region indices.

#### 2) get_otsu_regions (Identify Predicted Source Regions)
"def get_otsu_regions(out, labels, args_params=None)"
This function evaluates the neural network predictions and identifies which brain regions are predicted to be active.

The neural network outputs continuous activity values for all brain regions.
But for evaluation we need discrete active regions.

This function uses Otsu thresholding to separate active vs inactive regions.

Inputs
out
Predicted brain activity from the neural network.

Shape: (batch_size × time × brain_regions)

Example: (32 × 500 × 994)

Meaning:
- 32 samples in batch
- 500 time points
- 994 brain regions

labels
Ground truth source regions used during simulation.

Shape: (batch_size × num_sources × max_size)

These contain the true active brain regions.

args_params (optional) ,Extra parameters used to compute evaluation metrics.

Example:
dis_matrix : Distance matrix between brain regions.
Shape: 994 × 994

Processing Steps
###### Step 1: Normalize predicted activity
thre_source = np.abs(out[i])
thre_source = (thre_source - np.min(thre_source)) / np.max(thre_source)

This scales activity values between 0 and 1.

##### Step 2: Compute Otsu threshold
thresh = threshold_otsu(thre_source, nbins=100)
Otsu's method automatically finds a threshold separating low activity and high activity regions.

#### Step 3: Select regions above threshold
select_pixel = out[i] > thresh
This creates a binary mask of active regions.

##### Step 4: Identify active regions
otsu_region = np.where(np.sum(select_pixel, axis=0) > 7)[0]
Meaning: A region is considered active only if it stays active for more than 7 time points.

This avoids detecting random noise spikes.

Output stored
return_eval['all_regions'][i]
Predicted active brain regions.
return_eval['all_out'][i]
Predicted activity signals for those regions.

Evaluation Metrics (if args_params provided)
If a distance matrix is provided, the function calculates three evaluation metrics.

##### i) Precision 
precision = overlap_regions / predicted_regions
Meaning: Out of all predicted regions, how many were correct.

##### ii) Recall 
recall = overlap_regions / true_regions
Meaning: Out of all real active regions, how many were correctly detected.

##### iii) Localization Error (LE)
le_each_region = minimum distance between predicted and true regions
Then the average distance is computed.
