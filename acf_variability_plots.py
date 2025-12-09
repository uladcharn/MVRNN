import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

window = 50

fig_acf, axs_acf = plt.subplots(1, 4, figsize=(12, 3.5))
fig_rst, axs_rst = plt.subplots(1, 4, figsize=(12, 3.5))

datasets = ['DJI','NSDQ','AAPL','GOOGL']

for dataset, ax_acf, ax_rst in zip(datasets,axs_acf,axs_rst):
    # Loading data

    print(f'Loading {dataset} series...')

    series = pd.read_csv(f'./data/{dataset}.csv')
    series = series['x']

    print(f'Building ACF...')

    plot_acf(series, ax=ax_acf, lags=400)
    ax_acf.set_ylim(-0.4, 0.4)
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





