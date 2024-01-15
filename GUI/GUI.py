import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as fd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from sklearn.svm import SVR
from pyswarm import pso
import sys
import os

test_data = None


def process_signal():
    global test_data
    
    input_file = file_entry.get()
    if input_file == '':
        return

    progress_text.set('Processing the data...')

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

            # Normalize
            A = signal_num - np.mean(signal_num)
            B = (A - min(A)) / (max(A) - min(A))

            # Smoothen the signal using Savitzky-Golay filter
            smoothed_signal = savgol_filter(B, window_length=15, polyorder=2)

            # Perform Fast Fourier Transform (FFT)
            fft_signal = np.fft.fft(smoothed_signal)
            freq = np.fft.fftfreq(len(smoothed_signal))
            half_idx = len(smoothed_signal) // 2

            # Apply Butterworth low-pass filter
            order = 4  # Filter order
            cutoff_freq = 0.3  # Cutoff frequency in Hz
            nyquist_freq = 0.5 * Fs
            normal_cutoff = cutoff_freq / nyquist_freq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_signal = filtfilt(b, a, np.real(fft_signal))

            # Store the filtered data for the current column
            filtered_data.append(np.real(filtered_signal))

            # Create a figure with subplots for each stage of signal processing
            fig1, axs = plt.subplots(2, 1, figsize=(8, 6))
            fig1.suptitle('Signal Processing - Column {}'.format(col + 1))

            axs[0].plot(t, signal_num, '-r', linewidth=1.5)
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Amplitude (mV)')
            axs[0].set_title('Raw PEC Signal')
            axs[0].grid(True)

            axs[1].plot(t, B, '-r', linewidth=1.5)
            axs[1].grid(True)
            axs[1].set_xlabel('Time (ms)')
            axs[1].set_ylabel('Amplitude (mV)')
            axs[1].set_title('Normalized Signal with Offset Corrected')

            fig2, axs = plt.subplots(2, 1, figsize=(8, 6))
            fig2.suptitle('Signal Processing - Column {}'.format(col + 1))

            axs[0].plot(t, smoothed_signal, '-r', linewidth=1.5)
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Amplitude (mV)')
            axs[0].set_title('Smoothened Signal')
            axs[0].grid(True)

            axs[1].stem(freq[:half_idx], np.abs(fft_signal[:half_idx]), '-r')
            axs[1].set_xlabel('Frequency')
            axs[1].set_ylabel('Amplitude')
            axs[1].set_title('FFT of Truncated and Normalized Signal')
            axs[1].grid(True)

            # Create a separate figure for the filtered signal
            fig_filtered, ax_filtered = plt.subplots(figsize=(10, 5))
            ax_filtered.plot(t, filtered_signal, '-r', linewidth=1.5)
            ax_filtered.set_xlabel('Time (ms)')
            ax_filtered.set_ylabel('Amplitude (mV)')
            ax_filtered.set_title('Filtered Signal - Column {}'.format(col + 1))
            ax_filtered.grid(True)

            plt.show()

            # Store the filtered data for the current column
            filtered_data.append(np.real(filtered_signal))

        # Convert the filtered data to a DataFrame
        filtered_df = pd.DataFrame(filtered_data)
        test_data = filtered_df

        # Save the filtered data to a new Excel file for the current sheet
        output_file = sheet_name + '_filtered_data.xlsx'
        filtered_df.to_excel(output_file, index=False)

    progress_text.set('The data has been processed and has been saved.')


def characterize_signal():
    global test_data
    
    if probe_var.get() == 'Probe 1':
        train_data_file = 'C:/Users/HP/Desktop/FYP/Training Data/P1_Training_Data.xlsx'
    elif probe_var.get() == 'Probe 2':
        train_data_file = 'C:/Users/HP/Desktop/FYP/Training Data/P2_Training_Data.xlsx'
    elif probe_var.get() == 'Probe 3':
        train_data_file = 'C:/Users/HP/Desktop/FYP/Training Data/P3_Training_Data.xlsx'
    else:
        return

    progress_text.set('The PSO SVR is currently being trained...')

    # Load the training and testing data from Excel
    train_data = pd.read_excel(train_data_file)

    if test_data is None:
        progress_text.set('Please process the signal first.')
        return

    # Extract the input features and output labels for training and testing
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data

    # Define the objective function for PSO
    def objective_function(params, X, y):
        C, gamma = params
        svr = SVR(C=C, gamma=gamma, epsilon=0)
        svr.fit(X, y)
        y_pred = svr.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse

    # Define the bounds for C and gamma parameters
    lower_bound = [0.001, 0.001]
    upper_bound = [100, 100]

    # Run PSO optimization
    best_params, _ = pso(objective_function, lower_bound, upper_bound, args=(X_train, y_train))
    print(best_params)

    # Fit SVR model using the optimal parameters
    C_opt, gamma_opt = best_params
    svr = SVR(C=C_opt, gamma=gamma_opt, epsilon=0)
    svr.fit(X_train, y_train)
    print('Done training')

    progress_text.set('The depth of the crack is being characterized...')

    # Predict on the testing data
    y_pred = svr.predict(X_test)
    print('y prediction =', y_pred)

    progress_text.set('The depth of the crack is: {}'.format(y_pred))

    # Create a table to display the predictions
    result_table = ttk.Treeview(root)
    result_table['columns'] = ('Depth')
    result_table.column('#0', width=0, stretch=tk.NO)
    result_table.column('Depth', anchor=tk.CENTER, width=200)
    result_table.heading('Depth', text='Depth')

    # Insert each prediction into the table
    for i, pred in enumerate(y_pred):
        result_table.insert(parent='', index='end', text='', values=(pred,))


def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)


def exit_program():
    root.destroy()


# Create the GUI
root = tk.Tk()
root.title('Signal Processing')
root.geometry('600x400')

# Create and place the widgets
file_label = tk.Label(root, text='Enter Excel File Path:')
file_label.place(x=10, y=10)

file_entry = tk.Entry(root)
file_entry.place(x=150, y=10)

probe_label = tk.Label(root, text='Select Probe:')
probe_label.place(x=10, y=50)

probe_var = tk.StringVar(root)
probe_var.set('Probe 1')
probe_combobox = ttk.Combobox(root, textvariable=probe_var)
probe_combobox['values'] = ('Probe 1', 'Probe 2', 'Probe 3')
probe_combobox.place(x=150, y=50)

process_button = tk.Button(root, text='Process Signal', command=process_signal)
process_button.place(x=10, y=90)

characterize_button = tk.Button(root, text='Characterize Signal', command=characterize_signal)
characterize_button.place(x=10, y=130)

restart_button = tk.Button(root, text='Restart', command=restart_program)
restart_button.place(x=10, y=170)

exit_button = tk.Button(root, text='Exit', command=exit_program)
exit_button.place(x=10, y=210)

progress_text = tk.StringVar()
progress_text.set('')
progress_label = tk.Label(root, textvariable=progress_text)
progress_label.place(x=10, y=250)

root.mainloop()
