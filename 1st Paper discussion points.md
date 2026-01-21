### 							**1st Paper discussion points**

###### Step 1: Brain regions as possible sources
* First, the cortex is divided into 994 small regions — think of them as tiny patches of brain that could produce activity.
* These are all potential sources, but for any given EEG signal we usually assume only a few regions are “active” at a time, like in a real brain.

###### Step 2: Selecting active regions for each sample
* For every single training or testing example, we pick a subset of regions to be “active.”
  - Single-source data: pick 1 region randomly from the 994.
  - Two-source data: pick 2 regions randomly, which could be close or far apart.
* This simulates how in real EEG, usually only some brain areas are generating noticeable activity at a given time.
* Intuition: We are creating many “scenarios” where different brain regions are active. Each scenario becomes one training example.

###### Step 3: Generating time-series signals with Neural Mass Models (NMMs)
* Each active region uses a Jansen–Rit NMM, which is a mathematical model that can produce realistic brain signals over time.
* For each region:
  - The NMM generates a time series, basically a voltage signal that varies over time.
  - Parameters like amplitude, duration, and timing of spikes are randomly varied so each signal looks different, just like real EEG signals.

* Result: For each active region, we now have a simulated brain signal that represents how that region behaves electrically over time.

###### Step 4: Mapping brain signals to EEG channels (Leadfield Matrix)

* The leadfield matrix is a mathematical model that describes how electrical signals from the brain propagate through the skull and scalp to each EEG electrode.
* Using this matrix:
  - Each active region’s NMM signal is projected to all EEG electrodes.
  - Signals from multiple active regions combine linearly (like adding their effects at each electrode).

* Intuition:
  - The NMM signal is the source signal in the brain.
  - Applying the leadfield matrix transforms it into what an EEG cap would actually measure on the scalp.
  - This gives a synthetic EEG signal corresponding to the chosen active brain regions.


###### Step 5: Adding realistic noise
* EEG is always noisy in real life. To mimic this:
* Gaussian white noise is added to the scalp EEG signals.
* Different Signal-to-Noise Ratios (SNRs) like 5, 10, 15, 20 dB are used.
* This helps train the network to handle real EEG conditions, including noisy recordings.

DeepSIF Model Training
* The deep neural network architecture consists of two main modules:
  - Spatial Module: A five-layer fully connected network with skip connections, similar to a ResNet, which processes the spatial information from the EEG channels.
  - Temporal Module: Three Long Short-Term Memory (LSTM) layers that process the temporal dynamics of the signal.
* The model was trained using a Mean Square Error loss function and the Adam optimizer, with a weight decay of 1e-6 and a learning rate of 3e-4.

Discussion:
* DeepSIF is very good at locating brain activity, even with low-density EEG, which is better than traditional methods needing many electrodes.
* Its deep learning approach learns from a large simulated dataset, making it strong even with noisy or sparse EEG.
* Clinically, it can be used in routine EEG tests without expensive high-density caps or patient-specific MRIs.
* It is also faster than older methods, giving results quickly.

Challenges:
* Deep brain sources are still harder to localize, especially with very few electrodes.
* Separating sources that are very close together can be tricky in extreme cases, although DeepSIF handles this better than traditional methods.
* Sparse EEG (like 16 channels) is more likely to produce larger localization errors for deep sources.

Limitations:
* Only averaged interictal spikes were analyzed, not individual spikes.
* Subcortical (deep brain) structures were not included in the model.
* The surgical resection area used as ground truth is only an approximation of the actual seizure source.







