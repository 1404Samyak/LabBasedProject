## Training data 
- The .h5 file actually contains the EEG signals and .mat files contain information about those EEG signals 
- We created 640 simulations of EEG recordings which corresponds to 640 rows in .mat file 
- In one simulation 32 brain regions were active and 


- The shape of .h5 file is 500,76 means 500 timepoints and 76 active regions 

### Role of .mat file 
- The .mat file contains metadata for the simulations. It has 5 variables and each variable contains 640 entries since 640 simulations were generated.
#### i) selected_region : (640×1) Stores the seed brain region used for each simulation. This seed is used by the simulation algorithm to generate the active source patch (32 sources).
#### ii) nmm_idx (640×1) : Acts as an index pointer that indicates where the corresponding signal for that simulation is stored in
the .h5 dataset.
#### ii) sensor_snr (640×1) : Stores the signal-to-noise ratio used when adding noise to the EEG signals for each simulation.
#### iv) mag_change (640×1) : Represents the amplitude scaling factor applied to the simulated brain signals.
#### v) scale_ratio (640×1×2) : Stores the normalization parameters [min, max] used to scale the signals before they were used by the model.
