import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import pandas as pd
import models as models

from shapiro_wilk_tester import LatentNormalityTester

# Shapiro-Wilk Diagnostics

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--dataset', type=str, default='DJI',
                    help='The test dataset')
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

# Loading data

print('Loading validation data...')

validation_data = np.loadtxt(f"time_series_prediction/saved_validation_data/validate_x_{FLAGS.dataset}.csv")

validate_data = np.reshape(validation_data, (validation_data.shape[0],
                                             1, FLAGS.input_size)) # batch_size = 1

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

PATH = f"time_series_prediction/saved_models/{FLAGS.algorithm}_{FLAGS.dataset}"
model.load_state_dict(torch.load(PATH))
model.eval()

with torch.no_grad():
    _, _, info = model(torch.from_numpy(validate_data).float().to('cpu'), hidden_state=None, sample=False)

z_enc = info['all_z_enc'].reshape(-1, info['all_z_enc'].size(-1)).cpu().numpy()
z_prior = info['all_z_prior'].reshape(-1, info['all_z_prior'].size(-1)).cpu().numpy()

print('The information on encoder and prior collected')

print('Performing Shapiro-Wilk test for optimal mean transports...')
sw_tester = LatentNormalityTester(model, z_enc, z_prior)
print("\nStep 1: Collecting latent samples from validation set...")
z_enc_samples, z_prior_samples = sw_tester.collect_latent_samples(validate_data)
print('Step 2: Performing Shapiro-Wilk test on encoder latent codes...')
enc_results = sw_tester.shapiro_wilk_test(z_enc_samples, alpha=0.05)
print("\nStep 3: Performing Shapiro-Wilk test on prior samples (sanity check)")
prior_results = sw_tester.shapiro_wilk_test(z_prior_samples, alpha=0.05)
print("\nStep 4: Generating Report...")
sw_tester.generate_report(enc_results, prior_results, 
        save_path=f"time_series_prediction/results/shapiro_wilk_reports/sw_normality_report_{FLAGS.algorithm}_{FLAGS.dataset}_{FLAGS.K}_{FLAGS.lr}.txt")
print("\nStep 5: Generating visualizations...")

sw_tester.plot_latent_distributions(z_enc_samples, z_prior_samples, 
                                save_path=f'time_series_prediction/results/visualizations/latent_distributions_{FLAGS.algorithm}_{FLAGS.dataset}_{FLAGS.K}_{FLAGS.lr}.png')
sw_tester.plot_qq_plots(z_prior_samples, save_path=
                        f'time_series_prediction/results/visualizations/qq_plots_{FLAGS.algorithm}_{FLAGS.dataset}_{FLAGS.K}_{FLAGS.lr}.png')
print("\nStep 6: Generating Projections")
# canonical_proj = np.eye(FLAGS.latent_size)
random_projs = sw_tester.generate_random_projections()
# pca_proj = sw_tester.generate_pca_projections()
random_data_enc = z_enc_samples @ random_projs.T
random_data_prior = z_prior_samples @ random_projs.T
random_results_enc = sw_tester.test_projection_normality(random_data_enc)
random_results_prior = sw_tester.test_projection_normality(random_data_prior)

sw_tester.plot_p_value_distribution(
    random_results_enc,
    component = 'Encoder',
    save_path=f'time_series_prediction/results/visualizations/{FLAGS.dataset}_{FLAGS.algorithm}_{FLAGS.K}_{FLAGS.lr}_enc_pvalues.png'
)
sw_tester.plot_p_value_distribution(
    random_results_prior,
    component = 'Prior',
    save_path=f'time_series_prediction/results/visualizations/{FLAGS.dataset}_{FLAGS.algorithm}_{FLAGS.K}_{FLAGS.lr}_prior_pvalues.png'
)