import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import create_dataset, to_torch, detrend_local, bandpass_filter
import models as models
from statsmodels.tsa.seasonal import seasonal_decompose, STL

from shapiro_wilk_tester import LatentNormalityTester
from scipy.signal import hilbert

## initialization
PARSER = argparse.ArgumentParser()
# In time series prediction scenario, There are four dataset:
# 'tree7', 'traffic', 'arfima', 'DJI'.
PARSER.add_argument('--dataset', type=str, default='traffic',
                    help='The test dataset')
PARSER.add_argument('--algorithm', type=str, default='mVRNN_fixD',
                    help='The test algorithm')
PARSER.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
PARSER.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
PARSER.add_argument('--hidden_size', type=int, default=12,
                    help='Number of hidden units.')
PARSER.add_argument('--latent_size', type=int, default=6,
                    help='Number of latent units.')
PARSER.add_argument('--input_size', type=int, default=1,
                    help='Number of input units, as for time series it is 1.')
PARSER.add_argument('--output_size', type=int, default=1,
                    help='Number of output units, as for time series it is 1.')
PARSER.add_argument('--K', type=int, default=100,
                    help='Truncate the infinite summation at lag K.')
PARSER.add_argument('--patience', type=int, default=100, help='Patience.')
FLAGS = PARSER.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")

scaler = MinMaxScaler(feature_range=(0, 1))

def preprocess_data(df_data):
    # split train/val/test
    if FLAGS.dataset == 'AQ':
        train_size = 5000
        validate_size = 2500
    if FLAGS.dataset == 'DJI' or FLAGS.dataset == 'GOOGL':
        train_size = 2500
        validate_size = 1500
    if FLAGS.dataset == 'sunspots':
        train_size = 5000
        validate_size = 1300
    if FLAGS.dataset == 'ADU':
        train_size = 7000
        validate_size = 3000
    if FLAGS.dataset == 'AAPL' or FLAGS.dataset == 'META' or FLAGS.dataset == 'NSDQ':
        train_size = 1500
        validate_size = 500
    if FLAGS.dataset == 'traffic':
        train_size = 1200
        validate_size = 200
    if FLAGS.dataset == 'arfima_latent':
        train_size = 2000
        validate_size = 1200
    if FLAGS.dataset == 'GEOSD':
        train_size = 1200
        validate_size = 300
    if FLAGS.dataset == 'EEG':
        train_size = 4000
        validate_size = 2000
    if FLAGS.dataset == 'tree7':
        train_size = 2500
        validate_size = 1000
        
    batch_size = 1
    
    if FLAGS.dataset == 'GEOSD':
        series = pd.Series(df_data['mean'],index=df_data.index)
        series = pd.Series(detrend_local(series),index = series.index)
        x = np.array(series)
        y = np.array(series)
    elif FLAGS.dataset == 'EEG':
        series = pd.Series(df_data['Fp1'],index=df_data.index)
        filtered = bandpass_filter(series, fs=128, low=1, high=4)
        analytic = hilbert(filtered)
        log_amp = np.log(np.abs(analytic) + 1e-8)
        x = log_amp
        y = log_amp
    elif FLAGS.dataset == "AQ":
        df_data = df_data['C6H6(GT)']
        x = np.array(df_data)
        y = np.array(df_data)
    elif FLAGS.dataset == "ADU":
        df_data = df_data['sknt'].fillna(0)
        x = np.array(df_data)
        y = np.array(df_data)
    elif FLAGS.dataset == 'sunspots':
        series = pd.Series(df_data.iloc[:,4], index=df_data.iloc[:,4].index)
        # decomp = seasonal_decompose(series, period=365*10, extrapolate_trend='freq')
        # series_detrended = series - decomp.trend - decomp.seasonal
        p = 3895
        stl = STL(series, period=p, seasonal=31, trend=p+2, low_pass=p+2)
        res = stl.fit()
        series_detrended = series - res.trend
        # series = pd.Series(series_detrended.dropna(), index = series_detrended.index)
        x = series_detrended.to_numpy()
        y = series_detrended.to_numpy()
    else:
        x = np.array(df_data['x'])
        y = np.array(df_data['x'])
    x = x.reshape(-1, FLAGS.input_size)
    y = y.reshape(-1, FLAGS.output_size)
    # normalize the data
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
    # use this function to prepare the data for modeling
    data_x, data_y = create_dataset(x, y)

    # split into train and test sets
    train_x, train_y = data_x[0:train_size], data_y[0:train_size]
    validate_x, validate_y = data_x[train_size:train_size +
                                               validate_size], \
                             data_y[train_size:train_size +
                                               validate_size]
    
    # saving validation data for SW diagnostics
    PATH = f"time_series_prediction/saved_validation_data/validate_x_{FLAGS.dataset}.csv"
    np.savetxt(PATH, validate_x)

    test_x, test_y = data_x[train_size + validate_size:len(data_y)], \
                     data_y[train_size + validate_size:len(data_y)]

    # reshape input to be [time steps,samples,features]
    train_x = np.reshape(train_x,
                         (train_x.shape[0], batch_size, FLAGS.input_size))
    validate_x = np.reshape(validate_x, (validate_x.shape[0],
                                         batch_size, FLAGS.input_size))

    test_x = np.reshape(test_x,
                        (test_x.shape[0], batch_size, FLAGS.input_size))
    train_y = np.reshape(train_y,
                         (train_y.shape[0], batch_size, FLAGS.output_size))
    validate_y = np.reshape(validate_y, (validate_y.shape[0],
                                         batch_size, FLAGS.output_size))
    test_y = np.reshape(test_y, (test_y.shape[0],
                                 batch_size, FLAGS.output_size))
                                 
    return [train_x, train_y, validate_x, validate_y, test_x, test_y]

def train(train_data,model,criterion,optimizer,epoch):
    model.train()
    optimizer.zero_grad()
    train_x, train_y, validate_x, validate_y, test_x, test_y = train_data[0], train_data[1], train_data[2], train_data[3], train_data[4], train_data[5]
    target = torch.from_numpy(train_y).float().to(device)
    if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD']:
        output, hidden_state, info = model(torch.from_numpy(train_x).float().to(device))
        beta = min(1.0, epoch / FLAGS.K)
        loss = beta * model.get_kl_loss() + criterion(output, target)
        # nn.utils.clip_grad_norm_(model.parameters(), 10)
    elif FLAGS.algorithm in ['VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
        output, hidden_state, info = model(torch.from_numpy(train_x).float().to(device))
        z_t, z_prior_t = model.get_z_samples()
        mmd_penalty = model.mmd_penalty(z_t, z_prior_t)
        beta = min(10, 10 * epoch / FLAGS.K) 
        loss = criterion(output, target) + beta * mmd_penalty
    else:
        output, hidden_state = model(torch.from_numpy(train_x).float().to(device))
        loss = criterion(output, target)
    nn.utils.clip_grad_norm_(model.parameters(), 10)
    # validation - in terms of MSE 
    with torch.no_grad():
        if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD', 'VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
            val_y, _, info = model(torch.from_numpy(validate_x).float().to(device),
                         hidden_state, sample = False)
        else:
            val_y, _ = model(torch.from_numpy(validate_x).float().to(device),
                         hidden_state)
        target_val = torch.from_numpy(validate_y).float().to(device)
        val_loss = criterion(val_y, target_val)

    loss.backward()
    optimizer.step()
    if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD', 'VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
        return loss, val_loss, info
    else: 
        return loss, val_loss

def compute_test(best_model, train_data):
    model = best_model
    train_x, train_y, validate_x, validate_y, test_x, test_y = train_data[0], train_data[1], train_data[2], train_data[3], train_data[4], train_data[5]
    if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD', 'VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
        train_predict, hidden_state, _ = model(to_torch(train_x))
        val_predict, hidden_state, _ = model(to_torch(validate_x),
                                      hidden_state, sample=False)
        test_predict, _ , _ = model(to_torch(test_x), hidden_state, sample=False)
    else:
        train_predict, hidden_state = model(to_torch(train_x))
        val_predict, hidden_state = model(to_torch(validate_x),
                                      hidden_state)
        test_predict, _ = model(to_torch(test_x), hidden_state)
    train_predict = train_predict.detach().numpy()
    test_predict = test_predict.detach().numpy()
    # invert predictions
    test_predict_r = scaler.inverse_transform(test_predict[:, 0, :])
    test_y_r = scaler.inverse_transform(test_y[:, 0, :])
    # calculate error
    test_rmse = math.sqrt(mean_squared_error(test_y_r[:, 0],
                                             test_predict_r[:, 0]))
    test_mape = (abs((test_predict_r[:, 0] - test_y_r[:, 0]) /
                     test_y_r[:, 0])).mean()
    test_mae = mean_absolute_error(test_predict_r[:, 0],
                                   test_y_r[:, 0])

    if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD']:
        var_loss = model.get_kl_loss() + test_rmse**2
        return test_rmse, test_mape, test_mae, var_loss
    elif FLAGS.algorithm in ['VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
        z_t, z_prior_t = model.get_z_samples()
        mmd_penalty = model.mmd_penalty(z_t, z_prior_t)
        var_loss = test_rmse**2 + mmd_penalty
        return test_rmse, test_mape, test_mae, var_loss
    else:
        return test_rmse, test_mape, test_mae

def main():
    start = 0
    end = 10
    # read data
    # if FLAGS.dataset == 'sunspots':
    #     df_data = pd.read_csv('./data/time_series_prediction/' + FLAGS.dataset + '.csv', skiprows=67935, sep=";") 
    #     # Filter for 2004 to 2024
    #     df_data = df_data[(df_data.iloc[:,0] >= 2004) & (df_data.iloc[:,0] <= 2024)]
    if FLAGS.dataset == 'AQ':
        df_data = pd.read_csv(f'./data/time_series_prediction/{FLAGS.dataset}.csv', sep=";", decimal=',', skipfooter=114, engine='python')
        df_data = df_data.dropna(axis=1, how='all')
    elif FLAGS.dataset == 'ADU':
        df_data = pd.read_csv('./data/time_series_prediction/' + FLAGS.dataset + '.csv', usecols=['sknt'])
    else:
        df_data = pd.read_csv('./data/time_series_prediction/' + FLAGS.dataset + '.csv')
    train_data = preprocess_data(df_data)
    
    rmse_list = []
    mae_list = []
    var_list = []
    wae_list = []
    z_prior_wae = []
    z_enc_wae = []

    # SW diagnostics - model selection objects
    best_loss_global = np.inf
    best_model_global = None

    for i in range(start, end):
        seed = i
        print('seed ----------------------------------', seed)
        
        torch.manual_seed(seed)
        # initialize model
        if FLAGS.algorithm == 'RNN':
            model = models.RNN(input_size=FLAGS.input_size,
                               hidden_size=FLAGS.hidden_size,
                               output_size=FLAGS.output_size)
        elif FLAGS.algorithm == 'LSTM':
            model = models.LSTM(input_size=FLAGS.input_size,
                                hidden_size=FLAGS.hidden_size,
                                output_size=FLAGS.output_size)
        elif FLAGS.algorithm == 'GRU':
            model = models.GRU(input_size=FLAGS.input_size, 
                               hidden_size=FLAGS.hidden_size, 
                               output_size=FLAGS.output_size)
        elif FLAGS.algorithm == 'AttnLSTM':
            model = models.AttnLSTM(input_size=FLAGS.input_size, 
                                    hidden_size=FLAGS.hidden_size, 
                                    output_size=FLAGS.output_size)
        elif FLAGS.algorithm == 'mRNN_fixD':
            model = models.MRNNFixD(input_size=FLAGS.input_size,
                                    hidden_size=FLAGS.hidden_size,
                                    output_size=FLAGS.output_size,
                                    k=FLAGS.K)
        elif FLAGS.algorithm == 'mRNN':
            model = models.MRNN(input_size=FLAGS.input_size,
                                hidden_size=FLAGS.hidden_size,
                                output_size=FLAGS.output_size,
                                k=FLAGS.K)
        elif FLAGS.algorithm == 'mLSTM_fixD':
            model = models.MLSTMFixD(input_size=FLAGS.input_size,
                                     hidden_size=FLAGS.hidden_size,
                                     output_size=FLAGS.output_size,
                                     k=FLAGS.K)
        elif FLAGS.algorithm == 'mLSTM':
            model = models.MLSTM(input_size=FLAGS.input_size,
                                 hidden_size=FLAGS.hidden_size,
                                 output_size=FLAGS.output_size,
                                 k=FLAGS.K)
        elif FLAGS.algorithm == 'VRNN':
            model = models.VRNN(input_size=FLAGS.input_size,
                                hidden_size=FLAGS.hidden_size,
                                output_size=FLAGS.output_size,
                                latent_size=FLAGS.latent_size)
        elif FLAGS.algorithm == 'mVRNN':
            model = models.MVRNN(input_size=FLAGS.input_size,
                                hidden_size=FLAGS.hidden_size,
                                output_size=FLAGS.output_size,
                                latent_size=FLAGS.latent_size,
                                k=FLAGS.K)
        elif FLAGS.algorithm == 'mVRNN_fixD':
            model = models.MVRNNFixD(input_size=FLAGS.input_size,
                                hidden_size=FLAGS.hidden_size,
                                output_size=FLAGS.output_size,
                                latent_size=FLAGS.latent_size,
                                k=FLAGS.K)
        elif FLAGS.algorithm == 'mVRNN_WAE':
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
        else:
            print('Algorithm selection ERROR!!!')
        model.to(device)

        criterion = nn.MSELoss()
        criterion.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=FLAGS.lr)
        best_loss = np.inf
        best_train_loss = np.inf
        stop_criterion = 1e-5
        rec = np.zeros((FLAGS.epochs, 3))
        epoch = 0
        val_loss = -1
        train_loss = -1
        cnt = 0

        epoch_info = {
            'kl_loss': np.array(0),
            'nll_loss': np.array(0),
            'prior_mean': np.array(0),
            'prior_std': np.array(0),
            'enc_mean': np.array(0),  
            'enc_std': np.array(0),   
            'dec_mean': np.array(0),
            'dec_std': np.array(0)
        }

        while epoch < FLAGS.epochs:
            _time = time.time()
            if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD', 'VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
                loss, val_loss, info = train(train_data,model,criterion,optimizer,epoch)
                if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD']:
                    epoch_info['kl_loss'] = np.append(epoch_info['kl_loss'], info['kl_loss'].mean(dim=0).detach().cpu().numpy())
            else:
                loss, val_loss = train(train_data,model,criterion,optimizer,epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model = deepcopy(model)
            if best_loss < best_loss_global:
                best_model_global = best_model
                best_loss_global = best_loss
            # stop_criteria = abs(criterion(val_Y, target_val) - val_loss)
            if (best_train_loss - loss) > stop_criterion:
                best_train_loss = loss
                cnt = 0
            else:
                cnt += 1
            if cnt == FLAGS.patience:
                break
            # save training records
            time_elapsed = time.time()-_time
            rec[epoch, :] = np.array([loss.detach().cpu().numpy(), val_loss.detach().cpu().numpy(), time_elapsed])
            print("epoch: {:2.0f} train_loss - MSE (RNN/LSTM, mRNN/mLSTM) or Variational (VRNN/mVRNN): {:2.5f} val_loss (MSE): {:2.5f} "
                  "time: {:2.1f}s".format(epoch, loss.item(), val_loss.item(),
                                          time_elapsed))
            if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD', 'VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
                # Averages over batch
                prior_mean_t = np.mean(epoch_info['prior_mean'])
                prior_std_t  = np.mean(epoch_info['prior_std'])
                enc_mean_t   = np.mean(epoch_info['enc_mean'])
                enc_std_t    = np.mean(epoch_info['enc_std'])
                dec_mean_t   = np.mean(epoch_info['dec_mean'])
                dec_std_t    = np.mean(epoch_info['dec_std'])
                if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD']:
                    kl_loss_t = np.mean(epoch_info['kl_loss'])
                    nll_loss = np.mean(epoch_info['nll_loss'])
                    print("  " + "KL Loss: {:.6f}\t Gaussian NLL Loss: {:.6f}\t Prior Mean: {:.6f}\t Prior Std: {:.6f}\t Encoder Mean: {:.6f}\t Encoder Std: {:.6f}\t Decoder Mean: {:.6f}\t Decoder Std: {:.6f}".format(
                        kl_loss_t, nll_loss, prior_mean_t, prior_std_t, enc_mean_t, enc_std_t, dec_mean_t, dec_std_t
                        ))
                elif FLAGS.algorithm in ['VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
                    nll_loss = info['nll_loss'].mean().item()
                    print("  " + "Gaussian NLL Loss: {:.6f}\t Prior Mean: {:.6f}\t Prior Std: {:.6f}\t Encoder Mean: {:.6f}\t Encoder Std: {:.6f}\t Decoder Mean: {:.6f}\t Decoder Std: {:.6f}".format(
                        nll_loss, prior_mean_t, prior_std_t, enc_mean_t, enc_std_t, dec_mean_t, dec_std_t
                        ))
                    
            epoch += 1
        
        # make predictions
        if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD']:
            test_rmse, test_mape, test_mae, var_loss = compute_test(best_model,train_data)
        elif FLAGS.algorithm in ['VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
            test_rmse, test_mape, test_mae, wae_loss = compute_test(best_model,train_data)
        else:
            test_rmse, test_mape, test_mae = compute_test(best_model,train_data)

        rmse_list.append(test_rmse)
        mae_list.append(test_mae)
        print("epochs elapsed:{}".format(epoch))
        print('RMSE:{}'.format(rmse_list))
        print('MAE:{}'.format(mae_list))
        if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD']:
            var_loss = var_loss.detach().numpy()
            var_list.append(np.mean(var_loss))
            print("KL Loss:{}".format(var_list))
        elif FLAGS.algorithm in ['VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
            wae_loss = wae_loss.detach().numpy()
            wae_list.append(np.mean(wae_loss))
            print("WAE MMD Loss:{}".format(wae_list))

    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    mean_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)

    best_rmse = min(rmse_list)
    best_mae = min(mae_list)
        
    print("Mean - RMSE: " f"{mean_rmse}" " Standard Deviation - RMSE: " f"{std_rmse}")
    print("Mean - MAE: " f"{mean_mae}" " Standard Deviation - MAE: " f"{std_mae}")
    print("Best RMSE: " f"{best_rmse}")
    print("Best MAE: " f"{best_mae}")
    if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD']:
        mean_var_loss = np.mean(var_list)
        std_var_loss = np.std(var_list)   
        best_var_loss = min(var_list)
        print("Mean - Variational Loss: " f"{mean_var_loss}" " Standard Deviation - Variational Loss: " f"{std_var_loss}")
        print("Best Variational Loss: " f"{best_var_loss}")
    elif FLAGS.algorithm in ['VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
        mean_wae_loss = np.mean(wae_list)
        std_wae_loss = np.std(wae_list)   
        best_wae_loss = min(wae_list)
        print("Mean - WAE Loss: " f"{mean_wae_loss}" " Standard Deviation - WAE Loss: " f"{std_wae_loss}")
        print("Best WAE Loss: " f"{best_wae_loss}")

    # Saving Best Global Model

    PATH = f"time_series_prediction/saved_models/{FLAGS.algorithm}_{FLAGS.dataset}"
    torch.save(best_model_global.state_dict(), PATH)

    #Logging Forecasting Results

    log_dir = 'time_series_prediction/results/forecasting_metrics'
    log_file_name = f'{FLAGS.dataset}_{FLAGS.K}_{FLAGS.lr}'
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file_name)

    with open(file_path, 'a') as f:
        f.write(f"{FLAGS.algorithm}\n") 
        if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD']:
            metric_list = [mean_rmse,std_rmse,mean_mae,std_mae,mean_var_loss,std_var_loss, best_rmse,best_mae, best_var_loss]
            if FLAGS.dataset == 'DJI':
                metric_list = [m * 100 for m in metric_list]
            f.write("RMSE:{:.5f}, STD RMSE:{:.5f} ---- MAE:{:.5f}, STD MAE:{:.5f} ---- VAR LOSS:{:.5f}, STD VAR LOSS:{:.5f}"
            " ---- BEST RMSE:{:.5f}, BEST MAE:{:.5f}, BEST VAR LOSS:{:.5f}\n"
                    .format(*metric_list))
        elif FLAGS.algorithm in ['VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
            metric_list = [mean_rmse,std_rmse,mean_mae,std_mae,mean_wae_loss,std_wae_loss, best_rmse,best_mae, best_wae_loss]
            if FLAGS.dataset == 'DJI':
                metric_list = [m * 100 for m in metric_list]
            f.write("RMSE:{:.5f}, STD RMSE:{:.5f} ---- MAE:{:.5f}, STD MAE:{:.5f} ---- WAE LOSS:{:.5f}, STD WAE LOSS:{:.5f}"
            " ---- BEST RMSE:{:.5f}, BEST MAE:{:.5f}, BEST WAE LOSS:{:.5f}\n"
                    .format(*metric_list))
        else:
            metric_list = [mean_rmse,std_rmse,mean_mae,std_mae,best_rmse,best_mae]
            if FLAGS.dataset == 'DJI':
                metric_list = [m * 100 for m in metric_list]
            f.write("RMSE:{:.5f}, STD RMSE:{:.5f} ---- MAE:{:.5f}, STD MAE:{:.5f} ---- BEST RMSE:{:.5f}, BEST MAE:{:.5f}\n"
                .format(*metric_list))
        

if __name__ == "__main__":
    main()
    