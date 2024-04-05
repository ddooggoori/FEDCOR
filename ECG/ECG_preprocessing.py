import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, lfilter
import pywt
from pykalman import KalmanFilter
from scipy.signal import medfilt
import warnings
warnings.filterwarnings('ignore')



def ECG_maker(data, cut=None):
    
    """
    Preprocesses the input data by splitting it into segments.

    Parameters:
    - data: DataFrame - The input data to be preprocessed.
    - cut: str (default=None) - Specifies the duration of each segment ('1s', '2s', or '3s').

    Returns:
    - final_df: ndarray - The preprocessed data in the form of a 3D NumPy array.
    """
    
    final_set = []
    for i in range(len(data.columns)):
        # Extracting ECG signal data from DataFrame and converting it to a float type
        data_tmp = data.iloc[:, i].str.split(',', expand=True).astype(float)
        
        # Slicing the data based on the specified duration for each segment
        if cut == '1s':
            data_tmp = data_tmp.iloc[:, 500:4500]  # Exception 1s
        elif cut == '2s':
            data_tmp = data_tmp.iloc[:, 1000:4000]  # Exception 2s
        elif cut == '3s':
            data_tmp = data_tmp.iloc[:, 1500:3500]  # Exception 3s
        
        # Transposing the data for proper alignment
        data_tmp = data_tmp.T
        
        # Appending the preprocessed data to the final set
        final_set.append(data_tmp)

    # Converting the final set of preprocessed data into a 3D NumPy array
    arrays_df = [df.to_numpy() for df in final_set]
    final_df = np.stack(arrays_df, axis=-1)
    
    return final_df




def ECG_filtering(data, method='savgol', window_length=15, polyorder=2, 
                  cutoff_freq_lowpass=50, cutoff_freq_highpass=0.5, 
                  lowcut=0.5, highcut=50, fs=1000, wavelet='db4', wavelet_level=4,
                  median_kernel_size=3):
    """
    Apply ECG filtering to the input data.

    Parameters:
    - data: Input DataFrame containing ECG waveform data.
    - method: Filtering method ('savgol', 'lowpass', 'highpass', 'bandpass', 'wavelet', 'kalman', 'median', 'butter').
    - window_length: Window length for Savitzky-Golay filter.
    - polyorder: Polynomial order for Savitzky-Golay filter.
    - cutoff_freq_lowpass: Cutoff frequency for lowpass filter.
    - cutoff_freq_highpass: Cutoff frequency for highpass filter.
    - lowcut: Lower cutoff frequency for bandpass filter.
    - highcut: Upper cutoff frequency for bandpass filter.
    - fs: Sampling frequency.
    - wavelet: Wavelet type for Wavelet Transform.
    - wavelet_level: Level of decomposition for Wavelet Transform.
    - median_kernel_size: Kernel size for Median Filter.

    Returns:
    - Filtered DataFrame.
    """
    def apply_savgol_filter(data):
        return savgol_filter(data, window_length, polyorder)

    def apply_lowpass_filter(data):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq_lowpass / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, data)

    def apply_highpass_filter(data):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq_highpass / nyquist
        b, a = butter(4, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, data)

    def apply_bandpass_filter(data):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band', analog=False)
        return lfilter(b, a, data)

    def apply_wavelet_transform(data):
        coeffs = pywt.wavedec(data, wavelet, level=wavelet_level)
        coeffs[1:] = [pywt.threshold(c, value=0.1, mode='soft') for c in coeffs[1:]]
        reconstructed_data = pywt.waverec(coeffs, wavelet)
        return reconstructed_data

    def apply_kalman_filter(data):
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        filtered_state_means, _ = kf.filter(data)
        return filtered_state_means.flatten()

    def apply_median_filter(data):
        return medfilt(data, kernel_size=median_kernel_size)

    def apply_butterworth_filter(data):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band', analog=False)
        return lfilter(b, a, data)

    if method == 'savgol':
        filtered_data = data.apply(apply_savgol_filter)
    elif method == 'lowpass':
        filtered_data = data.apply(apply_lowpass_filter)
    elif method == 'highpass':
        filtered_data = data.apply(apply_highpass_filter)
    elif method == 'bandpass':
        filtered_data = data.apply(apply_bandpass_filter)
    elif method == 'wavelet':
        filtered_data = data.apply(apply_wavelet_transform)
    elif method == 'kalman':
        filtered_data = data.apply(apply_kalman_filter)
    elif method == 'median':
        filtered_data = data.apply(apply_median_filter)
    elif method == 'butter':
        filtered_data = data.apply(apply_butterworth_filter)
    else:
        raise ValueError("Invalid filtering method. Supported methods: 'savgol', 'lowpass', 'highpass', 'bandpass', 'wavelet', 'kalman', 'median', 'butter'.")

    return filtered_data



