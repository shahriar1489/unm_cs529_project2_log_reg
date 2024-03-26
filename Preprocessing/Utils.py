import librosa
import numpy as np
import os # used for iterating through file tree
import pandas as pd
#import math

#starts in directory where .au files are located. Runs process_au on each file. creates a single row vector containing target labels (i.e. "blues", etc.), combines all columns into one df, adds above row vector

#Given train, runs fxn on all subdirectories

def dir_to_df(subdir_path, train):
    
    # Directory path
    directory = subdir_path

    # Initialize an empty list to store file names
    au_files = []

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        # Check if the file has the ".au" extension
        if filename.endswith(".au"):
            # Append the file name to the list
            au_files.append(filename)
    


    col_vec = [] # contains feature data for each audio file in separate rows 
#    n = len(au_files)  # Length of the array
#    default_value = ""  # Default value for the array elements
#    col_vec = [default_value] * n

    # Process each audio file and append the feature data to col_vec
    #counter = 0
    for file in au_files:
        #if counter < 2:
        file_path = os.path.join(directory, file)
        col_vec.append(process_au(file_path, train))
            #counter = counter + 1

    # Convert col_vec into a single DataFrame
    df = pd.DataFrame(np.concatenate(col_vec, axis=1).T)
    print(df)
    return df

    # Concatenate the column vectors along the appropriate axis
    ##final_array = np.concatenate([np.pad(cv, ((0, col_length - cv.shape[0]), (0, 0)), mode='constant') for cv in col_vec], axis=1)

    # Print the list of .au files
    ##print(head(final_array))
    
    #return df











# Path to the audio file
#for i in range(101):  # Iterate from 0 to 100
    # Format the number with leading zeros to ensure it has 5 digits
 #   padded_number = str(i).zfill(5)
  #  audio_file_suffix = padded_number + ".au"
    #audio_file = 'resources/train/blues/blues.00000.au'

def get_interquartile_range(data):
  """
  data : numpy 2D matrix  
  """

  # get upper quartile 
  Q3 = np.percentile( data , 75, axis=1)

  # get lower quartile 
  Q1 = np.percentile(data, 25, axis=1)

  IQR =  Q3 - Q1 
  return IQR

def get_percentiles(data, percentiles): 
  """

  data : 2-D numpy array 
  percentiles : list w/ percentile values 
  
  Returns: 
    percentile across each row in the 2-D matrix 

  """
  results = []

  for p_ in percentiles: 

    results.append( np.percentile(data, p_, axis=1 ) )


  return tuple(results)

def process_au(au_path, train):
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

  # get minimum and maximum across each row
    min_mfcc = np.min(mfccs, axis=1)
    min_stft = np.min(stft, axis=1)
    min_chroma = np.min(chroma, axis=1)
    min_contrast = np.min(contrast, axis=1)
    min_zcr = np.min(zcr, axis=1)
    
    max_mfcc = np.max(mfccs, axis=1)
    max_stft = np.max(stft, axis=1)
    max_chroma = np.max(chroma, axis=1)
    max_contrast = np.max(contrast, axis=1)
    max_zcr = np.max(zcr, axis=1)
    
    
    # get median across each row 
    median_mfcc = np.median(mfccs, axis=1)
    median_stft = np.median(stft, axis=1)
    median_chroma = np.median(chroma, axis=1)
    median_contrast = np.median(contrast, axis=1)
    median_zcr = np.median(zcr, axis=1)
      

    # get interquartile range across each row 
    iq_mfcc = np.percentile( mfccs , 75, axis=1) - np.percentile( mfccs , 25, axis=1)
    iq_stft = np.percentile( stft , 75, axis=1) - np.percentile( stft , 25, axis=1)
    iq_chroma = np.percentile( chroma , 75, axis=1) - np.percentile( chroma , 25, axis=1)
    iq_contrast = np.percentile( contrast , 75, axis=1) - np.percentile( contrast , 25, axis=1)
    iq_zcr = np.percentile( zcr , 75, axis=1) - np.percentile( zcr , 25, axis=1)

    """
    NOTE: note taking the percentiles of stft.
    """


    # get percentiles across each row 
    percentiles = [12.5, 25.0, 37.5, 62.5, 75.0, 87.5]
    
    p_12_5_mfcc , p_25_mfcc, p_37_5_mfcc, p_62_5_mfcc , p_75_mfcc , p_87_5_mfcc = get_percentiles(mfccs, percentiles)
    p_12_5_chroma , p_25_chroma, p_37_5_chroma, p_62_5_chroma , p_75_chroma , p_87_5_chroma = get_percentiles(chroma, percentiles)
    p_12_5_constrast , p_25_contrast, p_37_5_contrast, p_62_5_contrast , p_75_contrast , p_87_5_contrast = get_percentiles(contrast, percentiles)
    p_12_5_zcr , p_25_zcr, p_37_5_zcr, p_62_5_zcr , p_75_zcr , p_87_5_zcr = get_percentiles(zcr, percentiles)

    # Concatenate the vectors
    concatenated_vectors = np.concatenate((mean_mfcc, mean_stft,
                                           mean_chroma, mean_contrast, mean_zcr,

                                           min_mfcc, #min_stft,
                                           min_chroma, min_contrast, min_zcr,

                                           max_mfcc, #max_stft,
                                           max_chroma, max_contrast, max_zcr,

                                           median_mfcc, #median_stft,
                                           median_chroma, median_contrast, median_zcr,

                                           iq_mfcc, #iq_stft,
                                           iq_chroma, iq_contrast, iq_zcr,

                                           p_12_5_mfcc , p_25_mfcc, p_37_5_mfcc, p_62_5_mfcc , p_75_mfcc , p_87_5_mfcc,
                                           p_12_5_chroma , p_25_chroma, p_37_5_chroma, p_62_5_chroma , p_75_chroma , p_87_5_chroma,
                                           p_12_5_constrast , p_25_contrast, p_37_5_contrast, p_62_5_contrast , p_75_contrast , p_87_5_contrast,
                                           p_12_5_zcr , p_25_zcr, p_37_5_zcr, p_62_5_zcr , p_75_zcr , p_87_5_zcr

                                           ), axis=0)
    #print("New_vector Shape:", concatenated_vectors.reshape(-1, 1).shape)


    if train:
        # Split the file path by "/"
        parts = audio_file.split("/")

        # Get the genre from the second-to-last part
        genre = parts[-2]
        # Convert genre to a NumPy array and reshape it to have a shape of (1, n_files)
        genre_array = np.array(genre).reshape(1, -1)

        # Add the genre row to concatenated_vectors
        concatenated_vectors_with_genre = np.vstack((genre_array, concatenated_vectors.reshape(-1, 1)))

    #print("New_vector Shape:", concatenated_vectors_with_genre.reshape(-1, 1).shape)
    #print(concatenated_vectors_with_genre[0])
    #print(concatenated_vectors.reshape(-1, 1))
    return concatenated_vectors.reshape(-1, 1)

dir_to_df("resources/train/blues/", True)
#process_au("resources/train/blues/blues.00000.au", True)
