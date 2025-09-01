import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import butter, lfilter
import numpy as np

# --- Data Loading & Stationarity Transformation ---
# Load the dataset
file_path = 'ep.csv'
try:
    df = pd.read_csv(file_path, index_col='DATE', parse_dates=True)
    value_column = 'IPG2211A2N'
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Apply seasonal and regular differencing to make the series stationary
df['stationary'] = df[value_column].diff(periods=12).diff()
stationary_series = df['stationary'].dropna()

# --- Task 3: Filtering Techniques on the Stationary Data ---
# We will create a new DataFrame to hold the filtered data for easy comparison.
filtered_df = pd.DataFrame(index=stationary_series.index)
filtered_df['Original Stationary'] = stationary_series

# 1. Apply a Simple Moving Average (SMA)
# A small window size is appropriate for smoothing noise
window_size = 5
sma = stationary_series.rolling(window=window_size, min_periods=1).mean()
filtered_df['SMA'] = sma

# 2. Apply Exponential Weighted Moving Average (EWMA)
ewma = stationary_series.ewm(span=window_size, adjust=False).mean()
filtered_df['EWMA'] = ewma

# 3. Apply a Butterworth Low-Pass Filter (smoothes the data, removes high-frequency noise)
def butter_lowpass_filter(data, cutoff, order):
    nyq = 0.5 * 1  # Assuming a sampling rate of 1 (since it's a time series)
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

order = 2  # Order of the filter
cutoff_low = 0.1 # Adjust this value to control the smoothing
low_pass_filtered = butter_lowpass_filter(stationary_series, cutoff=cutoff_low, order=order)
filtered_df['Low_Pass'] = low_pass_filtered

# Note: The high-pass filter is typically used to isolate noise.
# Since we have already isolated the noise by differencing, we will focus on low-pass filtering to smooth it.

# --- Task 4: Plot and Compare Series ---
plt.figure(figsize=(15, 10))
plt.plot(filtered_df.index, filtered_df['Original Stationary'], label='Original Stationary Series', color='blue', alpha=0.6)
plt.plot(filtered_df.index, filtered_df['SMA'], label=f'SMA (Window={window_size})', color='red', linestyle='--')
plt.plot(filtered_df.index, filtered_df['EWMA'], label=f'EWMA (Span={window_size})', color='green', linestyle='-.')
plt.plot(filtered_df.index, filtered_df['Low_Pass'], label='Butterworth Low-Pass', color='purple', linestyle='-')
plt.title('Original Stationary Series vs. Filtered Versions')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# --- Task 5: Re-compute the ACF of filtered data ---
print("\nComparing ACF of original and filtered stationary data...")
fig, axes = plt.subplots(3, 1, figsize=(15, 18))

# ACF of Original Stationary Data (should look like white noise)
plot_acf(stationary_series, ax=axes[0], lags=40, title='ACF of Original Stationary Data')

# ACF of SMA Filtered Data
plot_acf(filtered_df['SMA'].dropna(), ax=axes[1], lags=40, title='ACF of SMA Filtered Data')

# ACF of Low-Pass Filtered Data
plot_acf(filtered_df['Low_Pass'], ax=axes[2], lags=40, title='ACF of Low-Pass Filtered Data')

plt.tight_layout()
plt.show()