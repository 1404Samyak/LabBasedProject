# Training Data 
- Identify Possible Sources: They used the MRI scan to create a 3D model of the cortex.
- This model contains all possible points on the cortex where neuronal activity could originate. These are the candidate “sources” (15,002 grid points in their case).
- Create Activity Signals: From these possible sources, they randomly select 1–3 points to be active in a simulation.
- Each active source is assigned a simulated signal, which is a mix of sine waves (10–90 Hz) to mimic real brain activity.

- Map to MEG Sensors (Forward Mapping): Using the lead field matrix, they calculate what the MEG sensors would “see” if those sources were active.
- This basically transforms the source signals on the cortex into simulated MEG measurements at the 306 sensors.
- Add Noise: Gaussian noise is added to the sensor data to make it realistic, simulating the imperfect conditions of real MEG recordings.
- They first generated simulated MEG data as if it was recorded from a real human subject with a 306-sensor MEG system.

# Deep Learning in MEG 
- They developed and trained two distinct deep learning architectures to handle different types of MEG data inputs 
1) Deep MEG MLP(Multi Layer Perceptron)
- In DeepMEG-MLP, the network is designed to work with single-snapshot MEG data. A snapshot refers to the MEG measurements recorded at one specific time instant from all 306 sensors of the MEG system. 
- Thus, each input to the MLP is a vector of 306 values, representing the sensor readings at time t.
- During training, the model is not given an entire time series at once; instead, each time instant is treated as an independent training sample. A large number of such snapshots are generated from simulations, and each snapshot is paired with the corresponding ground-truth 3D source location. 
- By repeatedly feeding these single-snapshot inputs to the network, the MLP learns a direct mapping from instantaneous MEG sensor patterns to brain source coordinates. During real-time inference, this allows the model to localize brain activity at every millisecond (1 kHz) using only the current MEG measurement, without relying on past or future time information.

2) Deep MEG CNN
- In DeepMEG-CNN, the network is designed to use multiple MEG snapshots instead of a single time instant. Here, multiple snapshots mean that the input to the network consists of MEG sensor measurements collected over a short time window, rather than at just one moment. 
- Specifically, the input is a matrix of size 306 × T, where 306 corresponds to the number of MEG sensors and T represents the number of consecutive time samples. This allows the model to observe how MEG signals evolve over time, capturing temporal patterns such as oscillations, synchrony between sources, and phase relationships. 
- To process this time-series data efficiently, the CNN applies 1D convolutional filters along the time dimension, which act as space–time feature extractors. These filters learn meaningful temporal patterns while sharing parameters across time, resulting in fewer trainable parameters compared to fully connected networks. 
- By exploiting information from multiple snapshots, DeepMEG-CNN achieves more robust and accurate source localization, especially in noisy conditions or when multiple brain sources are active simultaneously.