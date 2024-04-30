# -*- coding: utf-8 -*-
import mne
import numpy as np
import matplotlib.pyplot as plt

# Lists to store ERPs for correct and incorrect sentences for all participants
all_erps_correct = []
all_erps_incorrect = []

# Loop through each subject's file, where X is the subject number 1-5
for subject_num in range(1, 6):
    file_name = f'SS0{subject_num}.cnt'
    
    # Load the EEG data for the subject
    raw = mne.io.read_raw_cnt(file_name, preload=True)
    raw.set_eeg_reference('average', projection=True)
    
    # Preprocess the data (filtering, finding events, epoching)
    raw.filter(l_freq=0.1, h_freq=30)
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id={'correct_grammar': 1, 'incorrect_grammar': 2},
                        tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)
    
    # Extract data from Cz for correct and incorrect sentences
    erp_correct = epochs['correct_grammar'].pick_channels(['Cz']).average()
    erp_incorrect = epochs['incorrect_grammar'].pick_channels(['Cz']).average()
    
    # Store the data for grand average calculation
    all_erps_correct.append(erp_correct.data)
    all_erps_incorrect.append(erp_incorrect.data)
    
    # If we are processing the first participant, plot their ERPs
    if subject_num == 1:
        times = epochs.times
        plt.figure(figsize=(10, 5))
        plt.plot(times, erp_correct.data.T, label='Correct Grammar')
        plt.plot(times, erp_incorrect.data.T, label='Incorrect Grammar')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(0, color='black', linestyle='--')  # Mark the onset of the error
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (µV)')
        plt.title(f'ERP from Cz Electrode - Participant {subject_num}')
        plt.legend()
        plt.show()

# Calculate the grand average across all participants
grand_average_correct = np.mean(np.concatenate(all_erps_correct, axis=0), axis=0)
grand_average_incorrect = np.mean(np.concatenate(all_erps_incorrect, axis=0), axis=0)

# Plot the grand average ERP
plt.figure(figsize=(10, 5))
plt.plot(times, grand_average_correct, label='Correct Grammar (All Participants)')
plt.plot(times, grand_average_incorrect, label='Incorrect Grammar (All Participants)')
plt.axhline(0, color='black', linestyle='--')
plt.axvline(0, color='black', linestyle='--')  # Mark the onset of the error
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('Grand Average ERP from Cz Electrode')
plt.legend()
plt.show()


