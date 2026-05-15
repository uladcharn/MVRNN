import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from copy import deepcopy
from utils import padding, accuracy
import review_classification.models as models
import torch.multiprocessing as mp


# initialization
PARSER = argparse.ArgumentParser()
# In review classification scenario,
# RNN, LSTM, mRNN_fixD, mLSTM_fixD, MVRNN_fixD are tested.
PARSER.add_argument('--algorithm', type=str, default='mRNN_fixD',
                    help='The test algorithm.')
PARSER.add_argument('--data', type=str, default='data',
                    help='The classification JSON data file.')
PARSER.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
PARSER.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
PARSER.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
PARSER.add_argument('--hidden_size', type=int, default=128,
                    help='Number of hidden units.')
PARSER.add_argument('--latent_size', type=int, default=32,
                    help='Number of latent units.')
PARSER.add_argument('--batch_size', type=int, default=64,
                    help='Number of batch size.')
PARSER.add_argument('--nb_class', type=int, default=5,
                    help='Number of class.')
PARSER.add_argument('--pad_size', type=int, default=150,
                    help='The padding size.')
PARSER.add_argument('--K', type=int, default=50,
                    help='Truncate the infinite summation at lag K.')
PARSER.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
FLAGS = PARSER.parse_args()

def preprocess_data(data_dict):
    # split train/val/test
    train_size = 282
    val_size = 50
    
    data = data_dict['data']
    label = np.array(data_dict['label'])
    length_list = np.array([len(term)-1 if len(term) < FLAGS.pad_size else
                           FLAGS.pad_size - 1 for term in data])
    input_size = len(data[0][0])
    data_pad = padding(data, FLAGS.pad_size, input_size)
    permutation = np.random.RandomState(seed=0).permutation(len(label))
    data_pad = data_pad[permutation]
    label = label[permutation]
    length_list = length_list[permutation]

    train_data = data_pad[0:train_size]
    train_label = label[0:train_size]
    train_length = length_list[0:train_size]

    val_data = data_pad[train_size:train_size+val_size]
    val_label = label[train_size:train_size+val_size]
    val_length = length_list[train_size:train_size+val_size]

    test_data = data_pad[train_size+val_size:]
    test_label = label[train_size+val_size:]
    test_length = length_list[train_size+val_size:]

    train_data = np.reshape(train_data, (train_data.shape[1],
                                         train_data.shape[0],
                                         train_data.shape[2]))
    val_data = np.reshape(val_data, (val_data.shape[1],
                                     val_data.shape[0],
                                     val_data.shape[2]))
    test_data = np.reshape(test_data, (test_data.shape[1],
                                       test_data.shape[0],
                                       test_data.shape[2]))

    train_data = torch.FloatTensor(train_data)
    train_label = torch.LongTensor(train_label)
    train_length = torch.LongTensor(train_length)

    val_data = torch.FloatTensor(val_data)
    val_label = torch.LongTensor(val_label)
    val_length = torch.LongTensor(val_length)

    test_data = torch.FloatTensor(test_data)
    test_label = torch.LongTensor(test_label)
    test_length = torch.LongTensor(test_length)

    return [(train_data,train_label,train_length), (val_data,val_label,val_length),
                (test_data,test_label,test_length), input_size]

def train(epoch, model, optimizer, batch_size, train_data_info, val_data_info):
        
        train_data, train_label, train_length = train_data_info[0], train_data_info[1], train_data_info[2]
        val_data, val_label, val_length = val_data_info[0], val_data_info[1], val_data_info[2]

        t_0 = time.time()
        total_batch = np.ceil(train_data.shape[1] / batch_size)
        loss_train_avg = 0.0
        acc_train_avg = 0.0

        for batch_num in range(int(total_batch)):
            if batch_num == total_batch - 1:
                batch_input = train_data[:, batch_num * batch_size:]
                batch_label = train_label[batch_num * batch_size:]
                batch_length = train_length[batch_num * batch_size:]
            else:
                batch_input = train_data[:, batch_num * batch_size:
                                            (batch_num + 1) * batch_size]
                batch_label = train_label[batch_num * batch_size:
                                            (batch_num + 1) * batch_size]
                batch_length = train_length[batch_num * batch_size:
                                            (batch_num + 1) * batch_size]
            model.train()
            optimizer.zero_grad()
            if FLAGS.algorithm in ['VRNN','mVRNN', 'mVRNN_fixD']:
                logit, info = model(batch_input, batch_length)
                beta = min(1.0, epoch / FLAGS.K)
                loss_train = beta * model.get_kl_loss() + F.nll_loss(logit, batch_label) 
            elif FLAGS.algorithm in ['mVRNN_WAE','mVRNN_fixD_WAE']:
                logit, info = model(batch_input, batch_length)
                z_t, z_prior_t = model.get_z_samples()
                mmd_penalty = model.mmd_penalty(z_t, z_prior_t)
                beta = min(10, 10 * epoch / FLAGS.K) 
                loss_train = F.nll_loss(logit, batch_label) + beta * mmd_penalty
            else:
                logit = model(batch_input, batch_length)
                loss_train = F.nll_loss(logit, batch_label) 
            acc_train = accuracy(logit, batch_label)
            loss_train_avg += loss_train.data.item()
            acc_train_avg += acc_train.item()
            loss_train.backward()
            optimizer.step()
        loss_train_avg = loss_train_avg / total_batch
        acc_train_avg = acc_train_avg / total_batch

        total_batch = np.ceil(val_data.shape[1] / batch_size)
        loss_val_avg = 0.0
        acc_val_avg = 0.0
        with torch.no_grad():
            for batch_num in range(int(total_batch)):
                if batch_num == total_batch - 1:
                    batch_input = val_data[:, batch_num * batch_size:]
                    batch_label = val_label[batch_num * batch_size:]
                    batch_length = val_length[batch_num * batch_size:]
                else:
                    batch_input = val_data[:, batch_num * batch_size:
                                                (batch_num + 1) * batch_size]
                    batch_label = val_label[batch_num * batch_size:
                                            (batch_num + 1) * batch_size]
                    batch_length = val_length[batch_num * batch_size:
                                                (batch_num + 1) * batch_size]
                if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD', 'VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
                    logit_val, info = model(batch_input, batch_length)
                else:
                    logit_val = model(batch_input, batch_length)
                loss_val = F.nll_loss(logit_val, batch_label)
                acc_val = accuracy(logit_val, batch_label)
                loss_val_avg += loss_val.data.item()
                acc_val_avg += acc_val.item()
            loss_val_avg = loss_val_avg/total_batch
            acc_val_avg = acc_val_avg / total_batch

        print('Train Stage, Epoch: {:04d}'.format(epoch + 1),
                'loss_train: {:.4f}'.format(loss_train_avg),
                'acc_train: {:.4f}'.format(acc_train_avg),
                'loss_val: {:.4f}'.format(loss_val_avg),
                'acc_val: {:.4f}'.format(acc_val_avg),
                'time_cost: {:.4f}s'.format(time.time()-t_0))
        
        if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD', 'VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
            return loss_val_avg, acc_val_avg, info
        else:
            return loss_val_avg, acc_val_avg

def compute_test(best_model, batch_size, test_data_info):
    # Restore best model
    model = best_model
    model.eval()

    test_data, test_label, test_length = test_data_info[0], test_data_info[1], test_data_info[2]

    total_batch = np.ceil(test_data.shape[1] / batch_size)
    loss_test_avg = 0.0
    acc_test_avg = 0.0
    f1_test_avg = 0.0
    pre_test_avg = 0.0
    recall_test_avg = 0.0
    with torch.no_grad():
        for batch_num in range(int(total_batch)):
            if batch_num == total_batch - 1:
                batch_input = test_data[:, batch_num * batch_size:]
                batch_label = test_label[batch_num * batch_size:]
                batch_length = test_length[batch_num * batch_size:]
            else:
                batch_input = test_data[:, batch_num * batch_size:
                                            (batch_num + 1) * batch_size]
                batch_label = test_label[batch_num * batch_size:
                                            (batch_num + 1) * batch_size]
                batch_length = test_length[batch_num * batch_size:
                                            (batch_num + 1) * batch_size]
            if FLAGS.algorithm in ['VRNN','mVRNN_fixD','mVRNN','mVRNN_fixD_WAE','mVRNN_WAE']:
                logit_test, info = model(batch_input, batch_length)
            else:
                logit_test = model(batch_input, batch_length)
            loss_test = F.nll_loss(logit_test, batch_label)
            acc_test = accuracy(logit_test, batch_label)
            pred = logit_test.max(1)[1].type_as(test_label).cpu().\
                detach().numpy()
            loss_test_avg += loss_test.data.item()
            acc_test_avg += acc_test.item()
            f1_test_avg += f1_score(test_label.cpu().detach().numpy(),
                                    pred, average='macro')
            pre_test_avg += precision_score(test_label.cpu().detach().
                                            numpy(), pred, average='macro')
            recall_test_avg += recall_score(test_label.cpu().detach().
                                            numpy(), pred, average='macro')
        loss_test_avg = loss_test_avg/total_batch
        acc_test_avg = acc_test_avg / total_batch
        f1_test_avg = f1_test_avg / total_batch
        pre_test_avg = pre_test_avg / total_batch
        recall_test_avg = recall_test_avg / total_batch

        print("Test set results:",
                "loss= {:.4f}".format(loss_test_avg),
                "accuracy= {:.4f}".format(acc_test_avg))
    acc_list.append(acc_test_avg)
    loss_list.append(loss_test_avg)
    f1_list.append(f1_test_avg)
    pre_list.append(pre_test_avg)
    recall_list.append(recall_test_avg)
    print(acc_list)
    print(loss_list)
    print(f1_list)
    print(pre_list)
    print(recall_list) 

    return acc_list, loss_list, f1_list, pre_list, recall_list

def run_seed(seed):
    torch.manual_seed(seed)
    start_time = time.time()

    with open(f'data/review_classification/{FLAGS.data}.json', 'r') as files:
        data_dict = json.load(files)

    train_data_info, val_data_info, test_data_info, input_size = preprocess_data(data_dict)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = FLAGS.batch_size
    hidden_size = FLAGS.hidden_size
    latent_size = FLAGS.latent_size
    nb_class = FLAGS.nb_class
    dropout = FLAGS.dropout
    k = FLAGS.K

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize model
    if FLAGS.algorithm == 'RNN':
        model = models.RNN(input_size, hidden_size, nb_class, dropout)
    elif FLAGS.algorithm == 'LSTM':
        model = models.LSTM(input_size, hidden_size, nb_class, dropout)
    elif FLAGS.algorithm == 'GRU':
        model = models.GRU(input_size, hidden_size, nb_class, dropout)
    elif FLAGS.algorithm == 'AttnLSTM':
        model = models.AttnLSTM(input_size, hidden_size, nb_class, dropout)
    elif FLAGS.algorithm == 'mRNN_fixD':
        model = models.MRNNFixD(input_size=input_size,
                                hidden_size=hidden_size,
                                output_size=nb_class,
                                k=k,
                                dropout=dropout)
    elif FLAGS.algorithm == 'mLSTM_fixD':
        model = models.MLSTMFixD(input_size=input_size,
                                    hidden_size=hidden_size,
                                    output_size=nb_class,
                                    k=k)
    elif FLAGS.algorithm == 'VRNN':
        model = models.VRNN(input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=nb_class,
                            latent_size=latent_size,
                            dropout=dropout)
    elif FLAGS.algorithm == 'mVRNN_fixD':
        model = models.MVRNNFixD(input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=nb_class,
                            latent_size=latent_size,
                            k=k,
                            dropout=dropout)
    elif FLAGS.algorithm == 'mVRNN':
        model = models.MVRNN(input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=nb_class,
                            latent_size=latent_size,
                            k=k,
                            dropout=dropout)
    elif FLAGS.algorithm == 'mVRNN_fixD_WAE':
        model = models.MVRNNFixD_WAE(input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=nb_class,
                            latent_size=latent_size,
                            k=k,
                            dropout=dropout)
        
    elif FLAGS.algorithm == 'mVRNN_WAE':
        model = models.MVRNN(input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=nb_class,
                            latent_size=latent_size,
                            k=k,
                            dropout=dropout)
    else:
        print('Algorithm selection ERROR!!!')
    model.to(device)

    optimizer = optim.AdamW(model.parameters(),
                               lr=FLAGS.lr,
                               weight_decay=FLAGS.weight_decay)
        
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

    if torch.cuda.is_available():
        model.cuda()
        train_data = train_data.cuda()
        train_label = train_label.cuda()
        val_data = val_data.cuda()
        val_label = val_label.cuda()
        test_data = test_data.cuda()
        test_label = test_label.cuda()
        test_length = test_length.cuda()
        train_length = train_length.cuda()
        val_length = val_length.cuda()

    loss_val_list = []
    best_loss = np.inf
    best_epoch = 0
    bad_counter = 0
    for epoch in range(100):
        if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD', 'VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
            loss_val_avg, acc_val_avg, info = train(epoch, model, optimizer, FLAGS.batch_size,
                        train_data_info, val_data_info)
            if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD']:
                epoch_info['kl_loss'] = np.append(epoch_info['kl_loss'], info['kl_loss'].mean(dim=0).detach().cpu().numpy())
        else:
            loss_val_avg, acc_val_avg = train(epoch, model, optimizer, FLAGS.batch_size,
                        train_data_info, val_data_info)
        if loss_val_avg < best_loss:
            best_loss = loss_val_avg
            best_epoch = epoch
            best_model = deepcopy(model)

    # Computing test
    print('Loading {}th epoch'.format(best_epoch))
    acc_list, loss_list, f1_list, pre_list, recall_list = compute_test(best_model,FLAGS.batch_size,test_data_info)

    elapsed_per_seed = time.time() - start_time
    # if FLAGS.algorithm in ['VRNN', 'mVRNN', 'mVRNN_fixD']:
    #     return {'seed': seed, 'epoch': epoch, 'elapsed_per_seed': elapsed_per_seed, 'acc': acc_list[0], 'acc_loss': loss_list[0], 'f1': f1_list[0],
    #             'precision': pre_list[0], 'recall': recall_list[0], 'best_loss': best_loss, 'best_model': best_model} # 
    # elif FLAGS.algorithm in ['VRNN_WAE', 'mVRNN_WAE', 'mVRNN_fixD_WAE']:
    #     return {'seed': seed, 'epoch': epoch, 'elapsed_per_seed': elapsed_per_seed, 'test_rmse':test_rmse, 'test_mae': test_mae, 'wae_loss': np.mean(wae_loss.detach().numpy()), 'best_loss': best_loss, 'best_model': best_model}
    # else:
    #     return {'seed': seed, 'epoch': epoch, 'elapsed_per_seed': elapsed_per_seed, 'test_rmse':test_rmse, 'test_mae': test_mae, 'best_loss': best_loss, 'best_model': best_model}
    return {'seed': seed, 'epoch': epoch, 'elapsed_per_seed': elapsed_per_seed, 'acc': acc_list[0], 'acc_loss': loss_list[0], 'f1': f1_list[0],
                'precision': pre_list[0], 'recall': recall_list[0], 'best_loss': best_loss, 'best_model': {k: v.detach() for k, v in best_model.state_dict().items()}}

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    seeds = range(0,30)

    # global parameters - do not work, use results
    acc_list = []
    loss_list = []
    f1_list = []
    pre_list = []
    recall_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    overall_start = time.time()

    print(f"Device used: {device}")
    
    # parallel processing
    results = {}
    with mp.Pool(5) as p:
        for out in tqdm(p.map(run_seed, seeds), total=len(seeds)):
            seed = out['seed']
            results[seed] = out
            print(f"seed {'-'*15} {seed}")
            print("epochs elapsed:{}".format(int(results[seed]['epoch'])))
            print('ACC:', results[seed]['acc'])
            print('LOSS:', results[seed]['acc_loss'])
            print('F1 :', results[seed]['f1'])
            print('Presicion: ', results[seed]['precision'])
            print('Recall :',results[seed]['recall'])

    best_seed, best_result = min(
        results.items(),
        key=lambda kv: kv[1]['best_loss']
    )

    best_loss_global = best_result['best_loss']
    best_model_global = best_result['best_model']

    print("="*50)
    print(f"\nTotal time: {time.time() - overall_start:.2f}s")
    print("="*50)
    print("Forecasting results are as follows:")
    print("="*50)

    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    max_acc = np.max(acc_list)
    mean_loss = np.mean(loss_list)
    std_loss = np.std(loss_list)
    min_loss = np.min(loss_list)
    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)
    max_f1 = np.max(f1_list)
    mean_pres = np.mean(pre_list)
    std_pres = np.std(pre_list)  
    max_pres = np.max(pre_list)
    mean_recall = np.mean(recall_list)
    std_recall = np.std(recall_list)
    max_recall = np.max(recall_list)

    print('ACC avg:', mean_acc, 'std:', std_acc,
          'max:', max_acc)
    print('LOSS avg:', mean_loss, 'std:', std_loss,
          'min', min_loss)
    print('F1 avg:', mean_f1, 'std:', std_f1,
          'max:', max_f1)
    print('Presicion avg:', mean_pres, 'std:', std_pres,
          'max:', max_pres)
    print('Recall avg:', mean_recall, 'std:', std_recall,
          'max:', max_recall)
    
    PATH = f"review_classification/saved_models/{FLAGS.algorithm}_{FLAGS.dataset}"
    torch.save(best_model_global.state_dict(), PATH)

    #Logging

    log_dir = 'results'
    log_file_name = f'{FLAGS.data}_{FLAGS.algorithm}_{FLAGS.K}_{FLAGS.lr}'
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file_name)

    with open(file_path, 'a') as f:
        f.write(f"{FLAGS.algorithm}\n")
        f.write("MEAN ACC:{:.5f}, STD ACC:{:.5f}, MAX ACC:{:.5f} ---- MEAN LOSS:{:.5f}, STD LOSS:{:.5f}, MIN LOSS:{:.5f} ---- MEAN F1:{:.5f}, STD F1:{:.5f}, BEST F1:{:.5f},\n"
                .format(mean_acc,std_acc,max_acc,mean_loss,std_loss,min_loss,mean_f1,std_f1,max_f1))
        f.write("MEAN PRES:{:.5f}, STD PRES:{:.5f}, MAX PRES:{:.5f} ---- MEAN REC:{:.5f}, STD REC:{:.5f}, MAX REC:{:.5f}\n"
                .format(mean_pres,std_pres,max_pres,mean_recall,std_recall,max_recall))