# Training data 
- The .h5 file actually contains the EEG signals and .mat files contain information about those EEG signals 
- We created 640 simulations of EEG recordings which corresponds to 640 rows in .mat file 
- In one simulation 32 brain regions were active and 
- The shape of .h5 file is 500,76 means 500 timepoints and 76 active regions 

### Role of .mat file 
- The .mat file contains metadata for the simulations. It has 5 variables and each variable contains 640 entries since 640 simulations were generated.

##### i) selected_region : (640×1) 
Stores the seed brain region used for each simulation. This seed is used by the simulation algorithm to generate the active source patch (32 sources).

##### ii) nmm_idx (640×1) : 
Acts as an index pointer that indicates where the corresponding signal for that simulation is stored in the .h5 dataset.

##### iii) sensor_snr (640×1) : 
Stores the signal-to-noise ratio used when adding noise to the EEG signals for each simulation.

##### iv) mag_change (640×1) : 
Represents the amplitude scaling factor applied to the simulated brain signals.

##### v) scale_ratio (640×1×2) : 
Stores the normalization parameters [min, max] used to scale the signals before they were used by the model.

# Training Part 

### 1) Loaders.py 
- This file converts our .mat and .h5 simulation files to training samples for the neural network 
- When training starts pytorch first creates a dataset here : `dataset = SpikeEEGBuild(data_root, fwd)`
- **class SpikeEEGBuild(Dataset)**: This class builds training samples 
- **def __init__(self, data_root, fwd, transform=None, args_params=None)**: This function runs once at the beginning which does the following tasks:

##### i) Load the .mat file
- `self.dataset_meta = loadmat(self.file_path)`
- This loads metadata of the .mat file containing following variables:
  - selected_region
  - nmm_idx
  - sensor_snr
  - mag_change
  - scale_ratio

##### ii) Determine the dataset size
- `self.dataset_len = self.dataset_meta['selected_region'].shape[0]`
- This gives the number of simulations in the dataset. In our case it becomes **640**, meaning there are **640 simulation samples**.

##### iii) Store the forward matrix
- `self.fwd = fwd`
- This forward matrix converts **brain source activity → EEG signals**. Shape of forward matrix is **76 × 994** (76 EEG sensors, 994 brain regions).

##### iv) Store scale ratio count
- `self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]`
- This tells how many possible scaling values exist for the signal amplitude.

---

### 2) __getitem__(index) (Most important function)
- This function runs every time PyTorch asks for a training sample. Example: `dataset[10]` -> This builds **one training example**.

##### i) Load the .h5 file (only first time)
`if self.data is None: self.data = h5py.File(base + '_nmm.h5', 'r')['data']`
- This loads the H5 dataset containing EEG signals. Shape of dataset: (1003, 500, 76). 
- Meaning: 1003 EEG samples, 500 time points, 76 EEG channels. Each sample therefore has shape: (500 × 76).

##### ii) Get active brain regions for this simulation
`raw_lb = self.dataset_meta['selected_region'][index]`
- This tells which brain regions were active in this simulation. Example: [120, 145, 200 ...]

##### iii) Remove padding values
`lb = raw_lb[~ispadding(raw_lb)]`
- Some arrays contain padding values used only for storage. These values are removed so only real active regions remain. 
- Example: Before cleaning: [120,145,200,-1,-1] and After cleaning: [120,145,200].

##### iv) Create empty brain activity matrix
`num_sources = self.fwd.shape[1]`
`raw_nmm = np.zeros((500, num_sources))`
- Example: (500 × 994). Meaning: 500 time points, 994 brain regions. Initially all regions have **zero activity**.

##### v) Loop through each source patch
`for kk in range(raw_lb.shape[0]):`
- Each simulation can contain multiple active source patches. The loader processes them one by one.

##### vi) Get region indices for the patch
`curr_lb = raw_lb[kk]`
- Example: [120,121,122,123] These neighboring regions form one **source patch**.

##### vii) Get corresponding EEG signal from H5
`current_nmm_data = self.data[self.dataset_meta['nmm_idx'][index][kk]]`
- nmm_idx tells which signal to fetch from the H5 dataset. 
- Example: nmm_idx = 25. So the loader fetches: self.data[25] of Shape: (500 × 76).

##### viii) Extract spike waveform
`ssig = current_nmm_data[:, [0]]`
- The first column is used as the reference spike waveform. Shape becomes: (500 × 1).

##### ix) Normalize signal
`ssig = ssig / (np.max(ssig) + 1e-9)`
- This scales the signal so the amplitude stays within a stable range.

##### x) Apply random amplitude scaling
`ssig *= scale_ratio[random_index]`
- The spike amplitude is randomly scaled using scale_ratio values. This introduces variability in signal strength. 
- Example scale ratios: 0.8, 1.0, 1.2.

##### xi) Apply spatial decay
`weight_decay = mag_change`
- Example: [1.0, 0.7, 0.4, 0.2]. Meaning: Center brain region → strongest signal, Neighboring regions → weaker signal.

##### xii) Insert signal into full brain activity
`raw_nmm[:, curr_lb] += ssig * weight_decay`
- The spike signal is placed into the full brain activity matrix. Only the active regions receive signals. 
- Now raw_nmm represents: Brain activity across **994 regions over time**.

##### xiii) Generate EEG signals using forward model
`eeg = self.fwd @ raw_nmm.T` (raw_nmm matrix after being transposed)
- Matrix multiplication: (Channels × Sources) × (Sources × Time). 
- Example: (76 × 994) × (994 × 500) which gives Result: (76 × 500). This produces the simulated **EEG signals**.

##### xiv) Add noise to EEG
`noisy_eeg = add_white_noise(eeg, csnr)`
- Noise level is taken from: sensor_snr. Example: 5 dB, 10 dB.

##### xv) Normalize EEG signal
Three steps are applied: 
- subtract channel mean 
- subtract time mean 
- divide by maximum amplitude 
This stabilizes the signal for neural network training.

##### xvi) Create ground truth source activity
`target = np.zeros_like(raw_nmm)`
`target[:, lb] = raw_nmm[:, lb]`
- Only active brain regions keep their signals. All other regions remain zero.

##### xvii) Normalize target
`target /= (np.max(target) + 1e-9)`
- Keeps target values within stable range.

##### xviii) Create final training sample
`sample = {'data': noisy_eeg, 'nmm': target, 'label': raw_lb, 'snr': csnr}`
- Meaning: data → EEG input signal, nmm → true brain source activity, label → indices of active regions, snr → noise level.

##### xix) Return sample
`return sample`
- The neural network receives: Input → EEG signal, Target → brain source activity.

---

### 4) Other classes in loaders.py
#### SpikeEEGLoad
- Used when data is already saved as individual .mat samples. It simply loads data instead of constructing it.
#### SpikeEEGBuildEval
- Similar to SpikeEEGBuild but used for evaluation/testing. Supports additional features like: realistic noise, frequency filtering.

---

# Network.py
This file defines the neural network architecture used to learn the mapping between EEG signals (from sensors) and brain source activity (in brain regions). The network is divided into two main stages:
1. Spatial Filtering → learns relationships between EEG sensors (76 eeg sensors)
2. Temporal Filtering → learns patterns over time,like EEG signals are time based signals 

The main model that combines both parts is called **TemporalInverseNet** (spatial filtering + temporal filtering both combined).

### 1) MLPSpatialFilter (Spatial Processing)
`class MLPSpatialFilter(nn.Module)`
This class performs spatial filtering of EEG signals across sensors (76 sensors) using fully connected layers (MLP). It learns how sensors interact with each other.It learns how EEG signals across 76 sensors are related to each other.

##### i) Initialization
`def __init__(self, num_sensor, num_hidden, activation):` Runs once when the model is created.
- **Parameters**: num_sensor (76), num_hidden (500), activation function (ReLU, Tanh etc.)
- **Layers**: fc11, fc12 (Linear num_sensor → num_sensor); fc21, fc22 (Linear num_sensor → num_hidden); fc23 (Linear num_sensor → num_hidden); value (Linear num_hidden → num_hidden).
- Means fc11+fc12 are two layers of one residual block and similarly  fc21+fc22+fc23 under one residual block 
- **Activation**: Created dynamically using `nn.__dict__[activation]()`.

##### ii) Forward Pass
`def forward(self, x):` Defines how EEG data flows.
- **Step 1**: First transformation with residual connection: `x = activation(fc12(activation(fc11(x))) + x)`. The `+ x` part is called a residual connection, which helps: better training, stable gradients.
- **Step 2**: Hidden feature transformation: `x = activation(fc22(activation(fc21(x))) + fc23(x))`.
- **Step 3**: Final spatial output: `out['value'] = value(x)`, `out['value_activation'] = activation(out['value'])`.

#### Following is the step-by-step flow:
##### 1. Initial Signal Refinement (Block 1)
- The process starts with the Input EEG data (B,76). This represents the electrical activity from 76 sensors.
- Local Processing: The data passes through fc11, an activation function, and then fc12. This stage focuses on learning local dependencies between the sensors.
- Linear Shortcut: While the data is being processed, a "Linear Shortcut" carries the original input forward and adds it to the output of fc12. This is the Residual Sum. It ensures that the network always has access to the raw sensor data, preventing the "vanishing gradient" problem.
- Output: After another activation, we get Spatial Features, still in the (B,76) dimension.

##### 2. Dimension Expansion (Block 2)
- Now that the spatial features are refined, the network needs to project them into a much higher-dimensional space to extract complex hidden patterns.
- Expansion: The features pass through fc21 and fc22, which expand the dimension from 76 to 500.
- Projection Shortcut: Because the dimension has changed, we can't simply add the original (B,76) input. Instead, we use fc23 as a "Projection Shortcut" to transform the 76-dimension features into 500-dimension features, which are then summed.
- Output: This results in the Hidden Representation (B,500).

##### 3. Final Output Generation
- The final stage prepares the data for the temporal (LSTM) part of the network.
- Value Layer: A final linear transformation (500→500) is applied.
- Dual-Path Output: One path goes directly to out['value']. This is a linear representation of the spatial data.
- The second path passes through a final Activation to become out['value_act'].
- By providing both a linear and a non-linear version of the spatial features, the network gives the subsequent LSTM layer a much richer set of information to work with for the final source localization.
- So we get two outputs one is raw features of shape B,500 and another is activated or non linear features of same shape B,500
but we feed only the activated/non linear features to the LSTM layer or the temporal filter 



### 2) TemporalFilter (Temporal Processing)
`class TemporalFilter(nn.Module)`
- So for the LSTM across each time step it receives B,500 so for 500 time points if we stack all we get a seq of shape B,500,500 which is fed as input to the LSTM layers
- The batch size is decided by the loaders.py not predefined 
This module learns temporal patterns in EEG signals using LSTM (Recurrent Neural Network) as EEG signals are time series data so we also need to learn how they vary over time
- For this reason the model uses an LSTM (Long Short-Term Memory network).LSTMs are designed to model sequences, remembering past information while processing future time steps.

##### i) Initialization
`def __init__(self, input_size, num_source, num_layer, activation):`
- **Parameters**: input_size (500), num_source (994) which is number of brain sources to estimate, num_layer basically number of stacked LSTM layers, activation.
- **LSTM creation**: `self.rnns.append(nn.LSTM(input_size, num_source, batch_first=True, num_layers=num_layer))`. Input: spatial features, Output: predicted brain source signals.

##### ii) Forward Pass
`def forward(self, x):`
- **Step 1**: Pass input through LSTM: `x, _ = l(x)`. The LSTM processes the temporal sequence.
- **Output**: `out['rnn'] = x`. Conceptual output shape: Batch × Time × BrainSources (Example: Batch × 500 time points × 994 brain regions).
- So the input sequence the LSTM gets is of shape B,500,76 only and there is only one LSTM layer so output after passing through LSTM layer is B,500,994 

### 3) TemporalInverseNet (Main Network)
`class TemporalInverseNet(nn.Module)`
Main model that combines spatial and temporal modules. Solves the EEG inverse problem: **EEG signals → Brain source activity**.
- **Initialization**: Creates Spatial Module (`MLPSpatialFilter`) and Temporal Module (`TemporalFilter`). `self.spatial = spat`.

# utils.py
Contains utility/helper functions for training, evaluation, and signal processing. Helps with padding, identifying predicted regions, adding noise, and mapping predictions.

#### 1) ispadding (Detect Padding Values)
`def ispadding(x)`
Identifies padding values in arrays that store brain region indices. Padding value is: **15213**.
- **How it works**: Step 1: Convert input to integer. Step 2: Compare values with 15213.
- **Example**: Input [120, 145, 200, 15213, 15213] → Output mask [False, False, False, True, True].
- **Purpose**: Helps remove padding values when extracting actual active brain regions.

#### 2) get_otsu_regions (Identify Predicted Source Regions)
`def get_otsu_regions(out, labels, args_params=None)`
Evaluates neural network predictions using Otsu thresholding to separate active vs inactive regions.

##### Inputs:
- **out**: Predicted brain activity (batch_size × time × brain_regions).
- **labels**: Ground truth source regions.
- **args_params**: Extra parameters like `dis_matrix` (994 × 994).

##### Processing Steps:
- **Step 1: Normalize**: Scales activity values between 0 and 1.
- **Step 2: Compute Otsu threshold**: `thresh = threshold_otsu(thre_source, nbins=100)`.
- **Step 3: Select regions**: Binary mask of active regions.
- **Step 4: Identify active regions**: A region is active if it stays active for **more than 7 time points** to avoid noise spikes.

##### Evaluation Metrics (if args_params provided):
- **i) Precision**: overlap_regions / predicted_regions (how many predictions were correct).
- **ii) Recall**: overlap_regions / true_regions (how many real active regions were detected).
- **iii) Localization Error (LE)**: Average minimum distance between predicted and true regions.

---

# main.py

This script serves as the primary execution engine. It configures hyperparameters, initializes the data pipeline, builds the neural network, and manages the training/validation loops.



### 1) Argument Parsing and Setup
`parser = argparse.ArgumentParser(description='DeepSIF Model')`
- **Purpose**: Defines user-controllable settings like `batch_size`, `lr` (learning rate), `epoch` (number of training rounds), and `device` (GPU/CPU).
- **Dynamic Dimensions**: `fwd_data = loadmat(...)['fwd']` loads the leadfield matrix to automatically detect `num_sensors` (76) and `num_regions` (994).

### 2) Data Loading Logic
- **DataLoaders**: Uses `DataLoader(train_data, batch_size=args.batch_size, shuffle=True)` to feed data into the model in small, shuffled groups (batches).
- **Pin Memory**: Set to `True` if using CUDA to speed up the transfer of data from CPU to GPU.

### 3) Model and Optimizer Initialization
- **Network Creation**: `net = network.__dict__[args.arch](...)` builds the `TemporalInverseNet` with specific parameters like `num_sensor=76` and `num_source=994`.
- **Optimizer**: `optim.Adam(net.parameters(), lr=args.lr)` sets up the Adam optimizer to update weights.
- **Criterion**: `torch.nn.MSELoss(reduction='sum')` defines the error metric—it calculates the difference between predicted brain activity and ground truth.

### 4) Resume and Checkpointing
`if args.resume:`
- **Logic**: Searches for a previous saved model (`epoch_X`). If found, it loads the `state_dict` (learned weights) and `optimizer` state to continue training exactly where it left off.
- **Strict Mode**: Uses `strict=True` to ensure the saved model architecture matches the current code perfectly.

### 5) The Training Function: `train()`
- **model.train()**: Puts the model in training mode (activates things like Dropout or BatchNorm if they were present).
- **Loss Calculation**: `out = model(data)['last']` gets the prediction, then `loss = criterion(out, nmm)` calculates how wrong it was.
- **Backpropagation**: `loss.backward()` calculates the gradients, and `optimizer.step()` updates the model weights to reduce the error.

### 6) The Validation Function: `validate()`
- **model.eval()**: Switches the model to evaluation mode.
- **torch.no_grad()**: Disables gradient calculation to save memory and speed up processing since we are only "testing," not "learning."

### 7) Result Storage
- **Logging**: Writes training/testing loss for every epoch into a `.log` file for later analysis.
- **Best Result**: `is_best = test_loss[-1] < best_result` checks if the current model is the most accurate one so far. If it is, it saves it as `model_best.pth.tar`.
- **Final Output**: `savemat(...)` saves the loss history as a `.mat` file for easy plotting in MATLAB.
