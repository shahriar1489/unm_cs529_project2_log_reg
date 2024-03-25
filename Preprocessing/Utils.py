import librosa
import numpy as np
import os # used for iterating through file tree
#import pandas as pd
#import math

def dir_to_df():
    #if train: # training data

    # Directory path
    directory = './'

    # Initialize an empty list to store file names
    au_files = []

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        # Check if the file has the ".au" extension
        if filename.endswith(".au"):
            # Append the file name to the list
            au_files.append(filename)

    # Print the list of .au files
    print(au_files)
    return au_files



# Path to the audio file
#for i in range(101):  # Iterate from 0 to 100
    # Format the number with leading zeros to ensure it has 5 digits
 #   padded_number = str(i).zfill(5)
  #  audio_file_suffix = padded_number + ".au"
    #audio_file = 'resources/train/blues/blues.00000.au'

def process_au(au_path):
    # Load the audio file using librosa
    audio_file = au_path
    audio_data, sample_rate = librosa.load(audio_file, sr=None)

    # Extract features using librosa
    # Example: compute the Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    stft = np.abs(librosa.stft(y=audio_data))
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(y=audio_data)

    # Calculate the mean across each row
    mean_mfcc = np.mean(mfccs, axis=1)
    mean_stft = np.mean(stft, axis=1)
    mean_chroma = np.mean(chroma, axis=1)
    mean_contrast = np.mean(contrast, axis=1)
    mean_zcr = np.mean(zcr, axis=1)

    # Reshape to a column vector
    mean_mfcc = mean_mfcc.reshape(-1, 1)
    mean_stft = mean_stft.reshape(-1, 1)
    mean_chroma = mean_chroma.reshape(-1, 1)
    mean_contrast = mean_contrast.reshape(-1, 1)
    mean_zcr = mean_zcr.reshape(-1, 1)

    # Concatenate the vectors
    concatenated_vectors = np.concatenate((mean_mfcc, mean_stft, mean_chroma, mean_contrast, mean_zcr), axis=0)

    # Print the shape of the MFCCs
    print("MFCCs Shape:", mean_mfcc.shape)
    print("MFCCs Shape:", mean_stft.shape)
    print("MFCCs Shape:", mean_chroma.shape)
    print("MFCCs Shape:", mean_contrast.shape)
    print("MFCCs Shape:", mean_zcr.shape)
    #print("New_vector Shape:", concatenated_vectors.shape)
    #print(mean_mfcc)
