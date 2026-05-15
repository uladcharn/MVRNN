#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it
# under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for
# more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import layers
from typing import Tuple, Optional, Dict

class RNN(nn.Module):
    """RNN model for review classification"""
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.rnn1 = nn.RNN(self.input_size, self.hidden_size,
                           dropout=dropout)
        self.rnn2 = nn.LSTM(self.hidden_size, self.hidden_size,
                            dropout=dropout)
        self.hidden2output = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_y, length, hidden_state=None):
        samples = self.dropout(input_y)
        rnn_out1, last_rnn_hidden = self.rnn1(samples)
        rnn_out1 = F.tanh(rnn_out1)
        rnn_out, last_rnn_hidden = self.rnn2(rnn_out1)
        rnn_out = F.tanh(rnn_out)
        rnn_out_sel = torch.cat([rnn_out[length[i], i].unsqueeze(0) for i in
                                 range(length.shape[0])], dim=0)
        output = self.hidden2output(rnn_out_sel)
        return F.log_softmax(output, dim=1)


class LSTM(nn.Module):
    """LSTM model for review classification"""
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(LSTM, self).__init__()
        self.in_size = in_size
        self.h_size = hidden_size
        self.out_size = out_size
        self.dropout = nn.Dropout(dropout)
        self.lstm_cell_1 = nn.LSTMCell(in_size, hidden_size)
        self.lstm_cell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, out_size)

    def forward(self, inputs, length, hx1=None, cx1=None, hx2=None, cx2=None):
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.h_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        for times in range(time_steps):
            if hx1 is None:
                hx1 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            if cx1 is None:
                cx1 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            inputs_t = self.dropout(inputs[times, :])
            hx1, cx1 = self.lstm_cell_1(inputs_t, (hx1, cx1))
            hx1 = F.tanh(hx1)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            hx2, cx2 = self.lstm_cell_2(hx1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2
        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.output(outputs_sel)
        return F.log_softmax(logit, dim=1)

class AttnLSTM(nn.Module):
    """AttnLSTM model for review classification"""
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(AttnLSTM, self).__init__()
        self.in_size = in_size
        self.h_size = hidden_size
        self.out_size = out_size
        self.dropout = nn.Dropout(dropout)
        self.lstm_cell_1 = nn.LSTMCell(in_size, hidden_size)
        self.lstm_cell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, out_size)
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

    def attention_net(self, lstm_output):
        if lstm_output.dim() == 2:
            lstm_output = lstm_output.unsqueeze(0) # Add batch dimension
        # Calculate attention scores
        attn_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        if attn_scores.dim() == 3:
            attn_scores = attn_scores.squeeze(-1)  # (batch, seq_len)
        elif attn_scores.dim() == 2:
            pass  # Already (batch, seq_len)
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)
        # Calculate context vector as weighted sum
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden)
        ).squeeze(1)  # (batch, hidden)
        return context

    def forward(self, inputs, length, hx1=None, cx1=None, hx2=None, cx2=None):
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.h_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        for times in range(time_steps):
            if hx1 is None:
                hx1 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            if cx1 is None:
                cx1 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            inputs_t = self.dropout(inputs[times, :])
            hx1, cx1 = self.lstm_cell_1(inputs_t, (hx1, cx1))
            context = self.attention_net(hx1)
            context = self.dropout(context)
            hx1 = F.tanh(hx1)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            hx2, cx2 = self.lstm_cell_2(hx1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2
        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.output(outputs_sel)
        return F.log_softmax(logit, dim=1)

class GRU(nn.Module):
    """GRU model for review classification"""
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(GRU, self).__init__()
        self.in_size = in_size
        self.h_size = hidden_size
        self.out_size = out_size
        self.dropout = nn.Dropout(dropout)
        self.gru1 = nn.GRUCell(in_size, hidden_size)
        self.gru2 = nn.LSTM(hidden_size, hidden_size,
                            dropout=dropout)
        self.output = nn.Linear(hidden_size, out_size)

    def forward(self, inputs, length, hx1=None, cx1=None, hx2=None, cx2=None):
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.h_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        for times in range(time_steps):
            if hx1 is None:
                hx1 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            inputs_t = self.dropout(inputs[times, :])
            hx1, _ = self.gru1(inputs_t)
            hx1 = F.tanh(hx1)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            hx2, _ = self.gru2(hx1)
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2
        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.output(outputs_sel)
        return F.log_softmax(logit, dim=1)

class MRNNFixD(nn.Module):
    """mRNN with fixed d for review classification"""
    def __init__(self, input_size, hidden_size, output_size, k,
                 dropout, bias=True):
        super(MRNNFixD, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bd1 = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.mrnn_cell_1 = layers.MRNNFixDCell(input_size, hidden_size,
                                               hidden_size, k)
        self.lstm_cell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_values):
        weights = torch.ones(self.k, 1, d_values.size(1),
                             dtype=d_values.dtype, device=d_values.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = \
                    self.get_ws(d_values[0, hidden].view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, length, hid=None, hx2=None, cx2=None):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)
        outputs = torch.Tensor(time_steps, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        self.d_values = 0.5 * F.sigmoid(self.bd1)
        weight_d = self.get_wd(self.d_values)

        for times in range(time_steps):
            temp = self.dropout(inputs[times, :])
            outputs1, hid = self.mrnn_cell_1(temp, weight_d, hid)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=inputs.dtype, device=inputs.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=inputs.dtype, device=inputs.device)
            hx2, cx2 = self.lstm_cell_2(outputs1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2

        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)
        return F.log_softmax(logit, dim=1)


class MLSTMFixD(nn.Module):
    """mLSTM with fixed d for review classification"""
    def __init__(self, input_size, hidden_size, k, output_size):
        super(MLSTMFixD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.b_d = Parameter(torch.Tensor(torch.zeros(1, hidden_size)),
                             requires_grad=True)
        self.output_size = output_size
        self.mlstm_cell = layers.MLSTMFixDCell(self.input_size,
                                               self.hidden_size,
                                               self.hidden_size,
                                               self.k)
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_values):
        weights = torch.ones(self.k, 1, d_values.size(1),
                             dtype=d_values.dtype, device=d_values.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = \
                    self.get_ws(d_values[0, hidden].view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, length, hidden=None, h_c=None, hx2=None,
                cx2=None):
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        self.d_values = 0.5 * F.sigmoid(self.b_d)
        weight_d = self.get_wd(self.d_values)
        for times in range(time_steps):
            outputs1, hidden, h_c = self.mlstm_cell(inputs[times, :], hidden,
                                                    h_c, weight_d)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=inputs.dtype, device=inputs.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=inputs.dtype, device=inputs.device)
            hx2, cx2 = self.lstm_cell(outputs1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2
        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)
        return F.log_softmax(logit, dim=1)
    
################ Variational Recurrent Neural Network ####################

class VRNN(nn.Module):
    """VRNN model for review classification"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, latent_size: int = None, dropout: float = None):
        super(VRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.latent_size = latent_size or hidden_size // 2
        
        # VRNN cell
        self.vrnn_cell = layers.VRNNCell(input_size, hidden_size, latent_size)

        # LSTM cell
        self.lstm = nn.LSTM(self.hidden_size + self.latent_size, self.hidden_size + self.latent_size,
                            dropout=dropout)
        
        #Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection layer
        self.hidden2output = nn.Linear(hidden_size + self.latent_size, output_size)
        
        # For storing KL losses during forward pass
        self.kl_losses = []
        
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state"""
        return torch.zeros(batch_size, self.hidden_size, device=device)
    
    def forward(self, inputs: torch.Tensor, length: torch.Tensor, hidden_state: Optional[torch.Tensor] = None,
                sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through VRNN for time series prediction
        
        Args:
            inputs: Input sequence, shape (seq_len, batch_size, input_size)
            hidden_state: Initial hidden state, shape (batch_size, hidden_size)
            sample: Whether to sample from latent distribution during training
            
        Returns:
            output: Predicted sequence, shape (seq_len, batch_size, output_size)
            last_hidden: Final hidden state, shape (batch_size, hidden_size)
        """
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        device = inputs.device
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden = self.init_hidden(batch_size, device)
        else:
            hidden = hidden_state
        
        # Clear KL losses for this forward pass
        self.kl_losses = []
        outputs = torch.Tensor(time_steps, batch_size, self.hidden_size+self.latent_size)
        
        for t in range(time_steps):
            # Get input at time t
            x_t = inputs[t]  # (batch_size, input_size)

            #Dropout (if applicable)

            x_t = self.dropout(x_t)
            
            # VRNN cell forward pass
            z_t, hidden, info = self.vrnn_cell(x_t, hidden, sample)

            z_t = F.tanh(z_t)
            
            # Store KL loss
            self.kl_losses.append(info['kl_loss'])
            
            # Generate output from hidden state and latent variable
            combined = torch.cat([hidden, z_t], dim=1)
            z_out, _ = self.lstm(combined)

            z_out = F.tanh(z_out)
            outputs[t, :] = z_out
        
        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)
        
        return F.log_softmax(logit,dim=1)
    
################ Memory-Augmented Variational Recurrent Neural Network ####################

class MVRNNFixD(nn.Module):
    """mRNN with fixed d for review classification"""
    def __init__(self, input_size, hidden_size, latent_size, output_size, k, dropout, bias=True):
        super(MVRNNFixD, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.bd1 = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.mvrnn_cell_1 = layers.MVRNNFixDCell(input_size, hidden_size, latent_size,
                                             hidden_size, k)
        
        self.lstm_cell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_value):
        weights = torch.ones(self.k, 1, d_value.size(1), dtype=d_value.dtype,
                             device=d_value.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = self.get_ws(d_value[0, hidden].
                                                         view([1]))
        return weights.squeeze(1)

    def forward(self, x, length, hid=None, hx2=None, cx2=None, sample = True):
        time_steps = x.size(0)
        batch_size = x.size(1)
        self.d_matrix = 0.5 * F.sigmoid(self.bd1)
        weights_d = self.get_wd(self.d_matrix)
        outputs = torch.Tensor(time_steps, batch_size, self.hidden_size)
        self.kl_losses = []
        self.nll_losses = []
        for times in range(time_steps):
            temp = self.dropout(x[times, :])
            outputs1, hid, info = self.mvrnn_cell_1(temp, weights_d, hid, sample = sample)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=x.dtype, device=x.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=x.dtype, device=x.device)
            hx2, cx2 = self.lstm_cell_2(outputs1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2

            self.kl_losses.append(info['kl_loss'])
            self.nll_losses.append(info['nll_loss'])

        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)
        return F.log_softmax(logit, dim=1), info
    
    def get_kl_loss(self) -> torch.Tensor:
        """Get total KL divergence loss for the sequence"""
        if not self.kl_losses:
            # Return a tensor on the right device with gradient tracking
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        # Stack preserves gradients - don't detach or use .item()
        return torch.stack(self.kl_losses).mean()
    
    def get_nll_loss(self) -> torch.Tensor:
        """Get total negative log-likelihood loss for the sequence"""
        if not self.nll_losses:
            # Return a tensor on the right device with gradient tracking
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        # Stack preserves gradients - don't detach or use .item()
        return torch.stack(self.nll_losses).mean()
    
class MVRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, latent_size, k, dropout, bias=True):
        super(MVRNN, self).__init__()

        self.x_dim = input_size
        self.h_dim = hidden_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.bd1 = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.mvrnn_cell_1 = layers.MVRNNCell(input_size, hidden_size, hidden_size, k, latent_size)

        self.lstm_cell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

        # self.hidden2output = nn.Linear(hidden_size + latent_size, output_size)

    def forward(self, x, length, hidden_state=None, hx2=None, cx2=None, sample=True):
        time_steps = x.size(0)
        batch_size = x.size(1)

        self.kl_losses = []
        self.nll_losses = []

        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=x.dtype, device=x.device)
        for times in range(time_steps):
            temp = self.dropout(x[times, :])
            outputs1, hidden_state, info = self.mvrnn_cell_1(temp, hidden_state, sample = sample)

            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=x.dtype, device=x.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=x.dtype, device=x.device)
            hx2, cx2 = self.lstm_cell_2(outputs1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2
            
            self.kl_losses.append(info['kl_loss'])
            self.nll_losses.append(info['nll_loss'])

        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)
        return F.log_softmax(logit, dim=1), info

    def get_kl_loss(self) -> torch.Tensor:
        """Get total KL divergence loss for the sequence"""
        if not self.kl_losses:
            # Return a tensor on the right device with gradient tracking
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        # Stack preserves gradients - don't detach or use .item()
        return torch.stack(self.kl_losses).mean()
    
    def get_nll_loss(self) -> torch.Tensor:
        """Get total negative log-likelihood loss for the sequence"""
        if not self.nll_losses:
            # Return a tensor on the right device with gradient tracking
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        # Stack preserves gradients - don't detach or use .item()
        return torch.stack(self.nll_losses).mean()
    
class MVRNNFixD_WAE(nn.Module):
    """mRNN with fixed d for review classification"""
    def __init__(self, input_size, hidden_size, latent_size, output_size, k, dropout, bias=True):
        super(MVRNNFixD_WAE, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.bd1 = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.mvrnn_cell_1 = layers.MVRNNFixDCell_WAE(input_size, hidden_size, latent_size,
                                             hidden_size, k)
        
        self.lstm_cell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_value):
        weights = torch.ones(self.k, 1, d_value.size(1), dtype=d_value.dtype,
                             device=d_value.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = self.get_ws(d_value[0, hidden].
                                                         view([1]))
        return weights.squeeze(1)

    def forward(self, x, length, hid=None, hx2=None, cx2=None, sample = True):
        time_steps = x.size(0)
        batch_size = x.size(1)
        self.d_matrix = 0.5 * F.sigmoid(self.bd1)
        weights_d = self.get_wd(self.d_matrix)
        outputs = torch.Tensor(time_steps, batch_size, self.hidden_size)
        self.all_z_enc = []
        self.all_z_prior = []
        self.nll_losses = []

        for times in range(time_steps):
            temp = self.dropout(x[times, :])
            outputs1, hid, info = self.mvrnn_cell_1(temp, weights_d, hid, sample = sample)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=x.dtype, device=x.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=x.dtype, device=x.device)
            hx2, cx2 = self.lstm_cell_2(outputs1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2

            self.all_z_enc.append(info['z_t'])
            self.all_z_prior.append(info['z_prior_t'])
            self.nll_losses.append(info['nll_loss'])

        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)

        info['all_z_enc'] = torch.stack(self.all_z_enc, dim=0)
        info['all_z_prior'] = torch.stack(self.all_z_prior, dim=0)

        return F.log_softmax(logit, dim=1), info 
    
    def rbf_kernel(self, x, y, sigma=1.0): 
        """Compute RBF kernel matrix between x and y"""
        # Expand for pairwise distances
        x = x.unsqueeze(1)  # [x_size, 1, dim]
        y = y.unsqueeze(0)  # [1, y_size, dim]
        
        # Squared distances
        dist = ((x - y) ** 2).sum(2)
        
        return torch.exp(-dist / (2 * sigma ** 2))
    
    def mmd_penalty(self, z_sample, z_prior, sigma=1.0):
        """Maximum Mean Discrepancy with RBF kernel"""
        k_zz = self.rbf_kernel(z_sample, z_sample, sigma)
        k_pp = self.rbf_kernel(z_prior, z_prior, sigma)
        k_zp = self.rbf_kernel(z_sample, z_prior, sigma)
        
        return k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()
    
    def get_nll_loss(self) -> torch.Tensor:
        """Get total negative log-likelihood loss for the sequence"""
        if not self.nll_losses:
            # Return a tensor on the right device with gradient tracking
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        # Stack preserves gradients - don't detach or use .item()
        return torch.stack(self.nll_losses).mean()
    
    def get_z_samples(self):
        """
        Get all z samples for MMD computation.
        
        Returns:
            z_enc: [seq_len, batch_size, z_dim] - samples from encoder
            z_prior: [seq_len, batch_size, z_dim] - samples from prior
        """
        if not self.all_z_enc or not self.all_z_prior:
            # Return empty tensors if no samples collected
            device = next(self.parameters()).device
            return (torch.zeros(0, 0, 0, device=device, requires_grad=True),
                    torch.zeros(0, 0, 0, device=device, requires_grad=True))
        
        # Stack WITHOUT taking mean - preserve all samples
        z_enc = torch.stack(self.all_z_enc, dim=0)      # [seq_len, batch_size, z_dim]
        z_prior = torch.stack(self.all_z_prior, dim=0)  # [seq_len, batch_size, z_dim]
        
        return z_enc, z_prior
    
class MVRNN_WAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, latent_size, k, dropout, bias=True):
        super(MVRNN_WAE, self).__init__()

        self.x_dim = input_size
        self.h_dim = hidden_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.bd1 = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.mvrnn_cell_1 = layers.MVRNNCell_WAE(input_size, hidden_size, hidden_size, k, latent_size)

        self.lstm_cell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

        # self.hidden2output = nn.Linear(hidden_size + latent_size, output_size)

    def forward(self, x, length, hidden_state=None, hx2=None, cx2=None, sample=True):
        time_steps = x.size(0)
        batch_size = x.size(1)

        self.all_z_enc = []
        self.all_z_prior = []
        self.nll_losses = []

        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=x.dtype, device=x.device)
        for times in range(time_steps):
            temp = self.dropout(x[times, :])
            outputs1, hidden_state, info = self.mvrnn_cell_1(temp, hidden_state, sample = sample)

            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=x.dtype, device=x.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=x.dtype, device=x.device)
            hx2, cx2 = self.lstm_cell_2(outputs1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2

            self.all_z_enc.append(info['z_t'])
            self.all_z_prior.append(info['z_prior_t'])
            self.nll_losses.append(info['nll_loss'])

        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)

        info['all_z_enc'] = torch.stack(self.all_z_enc, dim=0)
        info['all_z_prior'] = torch.stack(self.all_z_prior, dim=0)

        return F.log_softmax(logit, dim=1), info
    
    def rbf_kernel(self, x, y, sigma=1.0): 
        """Compute RBF kernel matrix between x and y"""
        # Expand for pairwise distances
        x = x.unsqueeze(1)  # [x_size, 1, dim]
        y = y.unsqueeze(0)  # [1, y_size, dim]
        
        # Squared distances
        dist = ((x - y) ** 2).sum(2)
        
        return torch.exp(-dist / (2 * sigma ** 2))
    
    def mmd_penalty(self, z_sample, z_prior, sigma=1.0):
        """Maximum Mean Discrepancy with RBF kernel"""
        k_zz = self.rbf_kernel(z_sample, z_sample, sigma)
        k_pp = self.rbf_kernel(z_prior, z_prior, sigma)
        k_zp = self.rbf_kernel(z_sample, z_prior, sigma)
        
        return k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()
    
    def get_z_samples(self):
        """
        Get all z samples for MMD computation.
        
        Returns:
            z_enc: [seq_len, batch_size, z_dim] - samples from encoder
            z_prior: [seq_len, batch_size, z_dim] - samples from prior
        """
        if not self.all_z_enc or not self.all_z_prior:
            # Return empty tensors if no samples collected
            device = next(self.parameters()).device
            return (torch.zeros(0, 0, 0, device=device, requires_grad=True),
                    torch.zeros(0, 0, 0, device=device, requires_grad=True))
        
        # Stack WITHOUT taking mean - preserve all samples
        z_enc = torch.stack(self.all_z_enc, dim=0)      # [seq_len, batch_size, z_dim]
        z_prior = torch.stack(self.all_z_prior, dim=0)  # [seq_len, batch_size, z_dim]
        
        return z_enc, z_prior
    
    def get_nll_loss(self) -> torch.Tensor:
        """Get total negative log-likelihood loss for the sequence"""
        if not self.nll_losses:
            # Return a tensor on the right device with gradient tracking
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        # Stack preserves gradients - don't detach or use .item()
        return torch.stack(self.nll_losses).mean()
