import pywt
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import numpy as np
import scipy as sp
from scipy import signal
from scipy.ndimage import uniform_filter1d
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.seasonal import STL

def detrend_linear(data):
    """Remove linear trend using scipy"""
    return signal.detrend(data, type='linear')

def detrend_moving_average(data, window=20):
    """Remove trend using moving average"""
    ma = uniform_filter1d(data, size=window, mode='nearest')
    return data - ma

def detrend_polynomial(data, degree=2):
    """Remove polynomial trend"""
    x = np.arange(len(data))
    coeffs = np.polyfit(x, data, degree)
    trend = np.polyval(coeffs, x)
    return data - trend

def detrend_differencing(data, order=1):
    """Remove trend using differencing"""
    detrended = np.copy(data)
    for _ in range(order):
        detrended = np.diff(detrended)
    return detrended

def detrend_local(x, window=400):
    """Local polynomial detrending (DFA-style)"""
    series = x.copy()
    for i in range(0, len(x), window):
        idx = slice(i, min(i+window, len(x)))
        t = np.arange(len(x[idx]))
        p = np.polyfit(t, x[idx], 1)
        series[idx] -= np.polyval(p, t)
    return series

def wavelet_decompose(x, wavelet='db4', level=7):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # coeffs = [A_L, D_L, D_{L-1}, ..., D1]
    return coeffs

from scipy.signal import butter, filtfilt, hilbert

def bandpass_filter(signal, fs=128, low=0.5, high=4):
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

datasets = ['DJI','AQ','ADU','arfima_latent']

window = 50

fig_acf, axs_acf = plt.subplots(1, len(datasets), figsize=(12, 3.5))
fig_rst, axs_rst = plt.subplots(1, len(datasets), figsize=(12, 3.5))

for dataset, ax_acf, ax_rst in zip(datasets,axs_acf,axs_rst):
    # Loading data

    print(f'Loading {dataset} series...')
    if dataset == 'sunspots':
        df = pd.read_csv(f'./data/time_series_prediction/{dataset}.csv', skiprows=67935, sep=";") # 67971 65701
        # Filter for 2004 to 2024
        series = df[(df.iloc[:,0] >= 2004) & (df.iloc[:,0] <= 2024)]
    elif dataset == 'AQ':
        series = pd.read_csv(f'./data/time_series_prediction/{dataset}.csv', sep=";", decimal=',', skipfooter=114, engine='python')
        series = series.dropna(axis=1, how='all')
    else:
        series = pd.read_csv(f'./data/time_series_prediction/{dataset}.csv')
    
    if dataset == "EEG": # figure out what feature to use
        # wavelet decomposition
        series = series['Fp1']
        filtered = bandpass_filter(series, fs=128, low=1, high=4)
        analytic = hilbert(filtered)
        log_amp = np.log(np.abs(analytic) + 1e-8)
        series = pd.Series(log_amp,index=series.index)
    elif dataset == "AQ":
        series = series['C6H6(GT)']
        series = series.loc[:9358]
        series = pd.Series(series, index = series.index)
    elif dataset == "WEATHER":
        series = pd.Series(series["Temperature"],index=series.index)
        # series = pd.Series(detrend_linear(series),index = series.index)
        series = np.log(series + 1e-8)
    elif dataset == 'ADU':
        series = series['sknt'].fillna(0)
        # series = pd.to_numeric(series, errors='coerce').dropna().values
        from hurst import compute_Hc
        H, c, data_hc = compute_Hc(series, kind='change', simplified=True)
        print(H)
    elif dataset == 'sunspots':
        # series_diff = series.diff(365)
        series = pd.Series(series.iloc[:,4], index=series.iloc[:,4].index)
        # decomp = seasonal_decompose(series, model='additive', period=4017, extrapolate_trend='freq')
        # decomp = seasonal_decompose(series, period=3835, extrapolate_trend='freq')
        # series_detrended = decomp.resid #  series - decomp.trend - decomp.seasonal
        p = 3895
        stl = STL(series, period=p, seasonal=31, trend=p+2, low_pass=p+2)
        res = stl.fit()

        series_detrended = res.resid
        series = pd.Series(series_detrended.dropna(), index = series_detrended.index)
        from hurst import compute_Hc
        H, c, data_hc = compute_Hc(series, kind='change', simplified=True)
        print(H)
    else:
        series = series['x'] # already preprocessed for long-range dependence in the form of abs-logs

    print(f'Building ACF...')

    if dataset == 'sunspots':
        plot_acf(np.abs(series), ax=ax_acf, lags=400)
    else:
        plot_acf(series, ax=ax_acf, lags=400)
    ax_acf.set_ylim(-0.2, 1)
    ax_acf.grid(True)
    ax_acf.set_title(f'{dataset}')

    print(f'Generating Rolling Statistics...')
    
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    ax_rst.plot(series, color='gray', alpha=0.5, label='Original')
    ax_rst.plot(rolling_mean, color='blue', label='Rolling Mean')
    ax_rst.plot(rolling_std, color='red', label='Rolling Std')
    ax_rst.set_title(f'{dataset}')
    ax_rst.legend()

# fig_acf.suptitle("Autocorrelation Plots")
# fig_acf.xlabel("Lags")
# fig_acf.ylabel("Autocorrelations")

fig_acf.tight_layout(rect=[0, 0, 1, 0.95])
fig_rst.tight_layout(rect=[0, 0, 1, 0.95])

print('Saving figures...')

fig_acf.savefig('acf_plots.png')
fig_rst.savefig('rolling_stat.png')





