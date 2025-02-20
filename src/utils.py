from sklearn.metrics import confusion_matrix
import urllib.request
from pyunpack import Archive


def display_progress_bar(progress, total_size, done=False):
    """Function responsible for displaying the progress bar.
    
    If done is True, shows the completed message instead of the progress.
    """
    if done:
        print(f"\r[{'=' * 50}] {total_size / (1024*1024):.2f} MB - Done!", end='\n')
    else:
        done_percentage = int(50 * progress / total_size)
        print(f"\r[{'=' * done_percentage}{' ' * (50 - done_percentage)}] {progress / (1024*1024):.2f}/{total_size / (1024*1024):.2f} MB", end='')
        

def print_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)

    print("\n📊 Confusion Matrix:\n")
    
    # Print header row
    header = "Predicted → | " + " | ".join([f"{name:^10}" for name in class_names])
    print(header)
    print("-" * len(header))

    # Print each row with true class labels
    for i, row in enumerate(cm):
        row_str = " | ".join([f"{num:^10}" for num in row])
        print(f"True {class_names[i]:<10} | {row_str} |")
    
    print("-" * len(header))


def download_file(url_base, url_suffix, output_path):
    print(f"Downloading the file: {os.path.basename(output_path)}")
    
    try:
        # Request the file size with a HEAD request
        req = urllib.request.Request(url_base + url_suffix, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])

        # Check if the file already exists and if not, download it
        if not os.path.exists(output_path):
            # Open the connection and the file in write-binary mode
            with urllib.request.urlopen(url_base + url_suffix) as response, open(output_path, 'wb') as out_file:
                block_size = 8192  # Define the block size for downloading in chunks
                progress = 0       # Initialize the progress counter
                
                # Download the file in chunks and write each chunk to the file
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    progress += len(chunk)
                    display_progress_bar(progress, file_size)  # Update the progress bar

            # After the download is complete, display final progress bar with "Download complete"
            display_progress_bar(progress, file_size, done=True)

            # Verify if the downloaded file size matches the expected size
            downloaded_file_size = os.stat(output_path).st_size
        else:
            downloaded_file_size = os.stat(output_path).st_size
        
        # If the file size doesn't match, remove the file and try downloading again
        if file_size != downloaded_file_size:
            os.remove(output_path)
            print("File size incorrect. Downloading again.")
            download_file(url_base, url_suffix, output_path)
    
    except Exception as e:
        print("Error occurs when downloading file: " + str(e))
        print("Trying to download again")
        download_file(url_base, url_suffix, output_path)


def extract_rar(file_path, output_dir): 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        Archive(file_path).extractall(output_dir)
        print(f"Extraction successful! Files extracted to: {file_path[:-4]}")
    except Exception as e:
        print(f"An error occurred during extraction: {str(e)}")


import h5py
import os
import numpy as np

def save_signal_hdf5(signal, label, h5_path="vibration_data.h5"):
    """
    Saves a signal iteratively in an HDF5 file along with its class label.

    Args:
        signal (np.ndarray): Vibration signal to save.
        label (int): Class label associated with the signal.
        h5_path (str): Path to the HDF5 file.
    """
    signal = np.array(signal, dtype=np.float32)  # Garantir que seja float32
    label = np.array([label], dtype=np.int32)    # Converter a classe para array NumPy

    # Criando dataset com estrutura correta
    with h5py.File(h5_path, "a") as f:
        if "signals" not in f:
            max_shape = (None, signal.shape[0])  # Permitir crescimento dinâmico
            f.create_dataset("signals", shape=(0, signal.shape[0]), maxshape=max_shape, dtype="float32", compression="gzip")

        if "labels" not in f:
            f.create_dataset("labels", shape=(0,), maxshape=(None,), dtype="int32")

        dataset_signals = f["signals"]
        dataset_labels = f["labels"]

        # Expandindo os datasets para armazenar a nova amostra
        current_size = dataset_signals.shape[0]
        dataset_signals.resize((current_size + 1, signal.shape[0]))
        dataset_labels.resize((current_size + 1,))

        # Adicionando a nova amostra
        dataset_signals[current_size] = signal
        dataset_labels[current_size] = label


import h5py
import numpy as np
import os

def save_mult_signal_hdf5(signals, labels, h5_path="vibration_data.h5"):
    """
    Saves one or multiple vibration signals iteratively in an HDF5 file along with their class labels.

    Args:
        signals (np.ndarray or list of np.ndarray): One or multiple vibration signals to save.
        labels (int or list of int): Class labels associated with the signals.
        h5_path (str): Path to the HDF5 file.
    """
    # Convert single signal to list if necessary
    if isinstance(signals, np.ndarray) and signals.ndim == 1:
        signals = [signals]
        labels = [labels]

    signals = np.array(signals, dtype=np.float32)  # Convert to NumPy array
    labels = np.array(labels, dtype=np.int32)      # Convert labels to NumPy array

    with h5py.File(h5_path, "a") as f:
        # Create dataset if it doesn't exist
        if "signals" not in f:
            max_shape = (None, signals.shape[1])  # Allow dynamic growth
            f.create_dataset("signals", shape=(0, signals.shape[1]), maxshape=max_shape, dtype="float32", compression="gzip")
        
        if "labels" not in f:
            f.create_dataset("labels", shape=(0,), maxshape=(None,), dtype="int32")

        dataset_signals = f["signals"]
        dataset_labels = f["labels"]

        # Get current dataset size
        current_size = dataset_signals.shape[0]

        # Resize datasets to accommodate new signals
        dataset_signals.resize((current_size + len(signals), signals.shape[1]))
        dataset_labels.resize((current_size + len(labels),))

        # Append new signals and labels
        dataset_signals[current_size:] = signals
        dataset_labels[current_size:] = labels


import pandas as pd
def save_signal_parquet(signal, parquet_path="vibration_data.parquet"):   
    df = pd.DataFrame({"signal": [signal.tolist()]})  # Store signal as a list
    if not os.path.exists(parquet_path):
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    else:
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", append=True)
