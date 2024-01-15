import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt

# Prompt the user for the input Excel file
input_file = input("Enter the signal path to the Excel file: ")

# Read the Excel file
xls = pd.ExcelFile(input_file)

# Iterate through each sheet in the Excel file
for sheet_name in xls.sheet_names:
    # Read the data from the current sheet
    data = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    num_columns = data.shape[1]

    # Generate time axis
    Fs = 100  # Sampling frequency
    T = 1 / Fs  # Sampling period
    L = data.shape[0]  # Length of signal
    t = np.arange(L) * T  # Time vector

    # Initialize a list to store filtered data for each column
    filtered_data = []

    # Process each column separately
    for col in range(num_columns):
        # Get the signal from the current column
        signal = data.iloc[:, col].values.flatten()

        # Convert signal to numeric data type
        signal_num = pd.to_numeric(signal, errors='coerce')

        # Smoothen the signal using Savitzky-Golay filter
        smoothed_signal = savgol_filter(signal_num, window_length=15, polyorder=2)

        # Apply Butterworth low-pass filter
        Fs = 100 # sampling frequency
        order = 4  # Filter order
        cutoff_freq = 1 # Cutoff frequency in Hz
        nyquist_freq = 0.5 * Fs
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, np.real(smoothed_signal))

        # Normalize
        A = (np.real(filtered_signal)) - np.mean((np.real(filtered_signal)))
        B = (A - min(A)) / (max(A) - min(A))

        # Perform Fast Fourier Transform (FFT)
        Fv = np.linspace(0, 1, int(L)) * nyquist_freq  # Frequency Vector
        Iv = np.arange(1, len(Fv))  # Index Vector
        X = np.fft.fft(B)

        # Store the filtered data for the current column
        filtered_data.append(np.abs(X[Iv]))

    # Convert the filtered data to a DataFrame
    filtered_df = pd.DataFrame(filtered_data)

    # Save the filtered data to a new Excel file for the current sheet
    output_file = sheet_name + '_filtered_data.xlsx'
    filtered_df.to_excel(output_file, index=False)

    print('Filtered data saved to', output_file)
