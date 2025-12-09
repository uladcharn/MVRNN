import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import pandas as pd
import models as models
import matplotlib.pyplot as plt

from shapiro_wilk_tester import LatentNormalityTester

# Shapiro-Wilk Diagnostics

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--algorithm', type=str, default='mVRNN_WAE',
                    help='The test algorithm')
PARSER.add_argument('--hidden_size', type=int, default=64,
                    help='Number of hidden units.')
PARSER.add_argument('--latent_size', type=int, default=16,
                    help='Number of latent units.')
PARSER.add_argument('--input_size', type=int, default=1,
                    help='Number of input units, as for time series it is 1.')
PARSER.add_argument('--output_size', type=int, default=1,
                    help='Number of output units, as for time series it is 1.')
PARSER.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
PARSER.add_argument('--K', type=int, default=100,
                    help='Truncate the infinite summation at lag K.')

FLAGS = PARSER.parse_args()

# Loading model

print('Loading model...')

if FLAGS.algorithm == 'mVRNN_WAE':
    model = models.MVRNN_WAE(input_size=FLAGS.input_size,
                        hidden_size=FLAGS.hidden_size,
                        output_size=FLAGS.output_size,
                        latent_size=FLAGS.latent_size,
                        k=FLAGS.K)
elif FLAGS.algorithm == 'mVRNN_fixD_WAE':
    model = models.MVRNNFixD_WAE(input_size=FLAGS.input_size,
                        hidden_size=FLAGS.hidden_size,
                        output_size=FLAGS.output_size,
                        latent_size=FLAGS.latent_size,
                        k=FLAGS.K)
    
datasets = ['DJI','NSDQ','AAPL','GOOGL']

fig, axs = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)
fig1, axs1 = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)

for dataset, ax, ax1 in zip(datasets,axs,axs1):

    # Loading data

    print(f'Loading validation data for {dataset}...')

    validation_data = np.loadtxt(f"time_series_prediction/saved_validation_data/validate_x_{dataset}.csv")

    validate_data = np.reshape(validation_data, (validation_data.shape[0],
                                                1, FLAGS.input_size)) # batch_size = 1
    
    PATH = f"time_series_prediction/saved_models/{FLAGS.algorithm}_{dataset}"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    with torch.no_grad():
        _, _, info = model(torch.from_numpy(validate_data).float().to('cpu'), hidden_state=None, sample=False)

    z_enc = info['all_z_enc'].reshape(-1, info['all_z_enc'].size(-1)).cpu().numpy()
    z_prior = info['all_z_prior'].reshape(-1, info['all_z_prior'].size(-1)).cpu().numpy()

    sw_tester = LatentNormalityTester(model, z_enc, z_prior)

    print("\nStep 1: Collecting latent samples from validation set...")
    z_enc_samples, z_prior_samples = sw_tester.collect_latent_samples(validate_data)

    print("\nStep 2: Generating Projections")

    random_projs = sw_tester.generate_random_projections()
    random_data_enc = z_enc_samples @ random_projs.T
    random_data_prior = z_prior_samples @ random_projs.T
    random_results_enc = sw_tester.test_projection_normality(random_data_enc)
    random_results_prior = sw_tester.test_projection_normality(random_data_prior)

    print("\nStep 3: Plotting histogram of p-values...")

    p_values_enc = np.array(random_results_enc['p_values'])

    ax.hist(p_values_enc, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, 
                label='α = 0.05')
    # Add text with normality rate
    normality_rate = random_results_enc['normality_rate'] * 100
    ax.set_title(f'{dataset}: {np.round(normality_rate,2)}%', weight='bold', fontsize = 14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    p_values_prior = np.array(random_results_prior['p_values'])

    ax1.hist(p_values_prior, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, 
                label='α = 0.05')

    normality_rate = random_results_prior['normality_rate'] * 100
    ax1.set_title(f'{dataset}: {np.round(normality_rate,2)}%', weight='bold', fontsize = 14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0.05, 1, 1])
fig1.tight_layout(rect=[0, 0.05, 1, 1])

save_path = f"time_series_prediction/results/visualizations/{FLAGS.algorithm}_{FLAGS.K}_{FLAGS.lr}_enc_pvalues_grouped.png"
save_path2 = f"time_series_prediction/results/visualizations/{FLAGS.algorithm}_{FLAGS.K}_{FLAGS.lr}_prior_pvalues_grouped.png"

fig.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved p-value distribution for encoder samples to: {save_path}")
fig1.savefig(save_path2, dpi=300, bbox_inches='tight')
print(f"Saved p-value distribution for prior samples to: {save_path}")
plt.close()
