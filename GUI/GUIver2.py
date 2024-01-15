import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as fd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
import sys
import os
import joblib

test_data = None
result_table = None


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

            # Smoothen the signal using Savitzky-Golay filter
            smoothed_signal = savgol_filter(signal_num, window_length=15, polyorder=2)

            # Apply Butterworth low-pass filter
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
            Fv = np.linspace(0, nyquist_freq, len(B))  # Frequency Vector with the same length as B # Frequency Vector
            Iv = np.arange(1, len(Fv))  # Index Vector
            X = np.fft.fft(B)

            # Store the filtered data for the current column
            filtered_data.append(np.abs(X[Iv]))

            # Create a figure with subplots for each stage of signal processing
            fig1, axs = plt.subplots(2, 1, figsize=(6, 8))
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
            axs[1].set_title('Filtered Normalized Signal')
            
            # Create a separate figure for the FFT signal
            fig_filtered, ax_filtered = plt.subplots(figsize=(10, 5))
            ax_filtered.stem(Fv[Iv], np.abs(X[Iv]), '-r')
            ax_filtered.set_xlabel('Time (ms)')
            ax_filtered.set_ylabel('Amplitude (mV)')
            ax_filtered.set_title('Fast Fourier Transform Amplitude Spectrum - Column {}'.format(col + 1))
            ax_filtered.grid(True)

            # Create figure for zoom-in FFT signal
            f_min = 0  # Minimum frequency (in Hz)
            f_max = 4  # Maximum frequency (in Hz)
            Iv_filtered = np.where((Fv >= f_min) & (Fv <= f_max))

            fig_fft, ax_fft = plt.subplots(figsize=(10, 5))
            ax_fft.stem(Fv[Iv_filtered], np.abs(X[Iv_filtered]), '-r')
            ax_fft.set_xlabel('Frequency (Hz)')
            ax_fft.set_ylabel('Amplitude')
            ax_fft.set_title('Fast Fourier Transform Amplitude Spectrum - Column {}'.format(col + 1))
            ax_fft.grid(True)
            ax_fft.set_xlim(f_min, f_max)

            plt.show()

        # Convert the filtered data to a DataFrame
        test_data = pd.DataFrame(filtered_data)

        # Save the filtered data to a new Excel file for the current sheet
        output_file = sheet_name + '_filtered_data.xlsx'
        test_data.to_excel(output_file, index=False)

        progress_text.set('The has been saved and loaded')


def characterize_signal():
    global test_data, result_table

    if probe_var.get() == 'Probe 1':
        loaded_model = joblib.load('P1_best_model.joblib')
    elif probe_var.get() == 'Probe 2':
        loaded_model = joblib.load('P2_best_model.joblib')
    elif probe_var.get() == 'Probe 3':
        loaded_model = joblib.load('P3_best_model.joblib')
    else:
        return

    progress_text.set('The PSO SVR is currently being trained...')

    if test_data is None:
        progress_text.set('Please process the signal first.')
        return

    # Predict on the data
    y_pred = loaded_model.predict(test_data)

    progress_text.set('The depth of the crack is being characterized...')

    progress_text.set('The depth of the crack is: {} mm'.format(y_pred))

    # Delete the existing table (if any)
    if result_table:
        result_table.destroy()

    # Create a new table to display the predictions
    result_table = ttk.Treeview(root)
    result_table['columns'] = ('Depth')
    result_table.column('#0', width=0, stretch=tk.NO)
    result_table.column('Depth', anchor=tk.CENTER, width=200)
    result_table.heading('Depth', text='Depth (mm)')

    # Insert each prediction into the table
    for i, pred in enumerate(y_pred):
        result_table.insert(parent='', index='end', text='', values=(pred,))

    # Place the table in the GUI
    result_table.place(x=10, y=300)


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
