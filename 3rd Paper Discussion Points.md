# Introduction 
- Same for all papers 

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

# Method
- The DeepMEG models were trained using Stochastic Gradient Descent (SGD), which is a common optimization method in deep learning. 
- In SGD, the network is repeatedly shown training examples, compares its predicted brain source location with the true location, and then slightly adjusts its internal parameters to reduce the error. 
- The error is measured using the Mean Squared Error (MSE) loss function, which penalizes the network based on how far its predicted 3D coordinates are from the true source coordinates in the simulated data. 
- This encourages the model to produce source estimates that are as close as possible to the ground truth.
- To ensure robustness and generalization, the models were trained on data generated with different noise levels (varying SNRs) and different inter-source correlations, allowing the network to learn to localize brain activity accurately even when the MEG signals are noisy or when multiple brain regions are active simultaneously.

# Results
- Baseline Comparison: The DeepMEG models were evaluated against the traditional RAP-MUSIC MEG source localization method.
- Higher Accuracy: DeepMEG achieved better localization accuracy, especially in difficult conditions such as low SNR and highly correlated (synchronous) brain sources.

- Much Faster Computation: DeepMEG was approximately 10,000× faster than RAP-MUSIC, estimating source locations in less than 0.2 ms, enabling 1 kHz real-time imaging.
- Robustness to Head Movement: DeepMEG remained stable even when small head rotations (1°) or translations (3 mm) were introduced, whereas traditional methods showed significant performance degradation.

# Limitations and Future Prospects
- Subject Specificity: At present, each DeepMEG model must be trained separately for every individual because brain anatomy (shape and size of the cortex) varies from person to person.
- Sensitivity to Assumptions: The model’s accuracy depends on how well the simulated training data reflects real brain activity, including assumptions about the number of active sources and the type and level of noise.
- Future Direction – Transfer Learning: The authors propose using transfer learning, where a model trained on one subject can be quickly fine-tuned for a new subject, reducing training time and data requirements.
- Future Direction – More Realistic Simulations: Further research is needed to generate more realistic simulated data so that the model generalizes better to real MEG experiments.