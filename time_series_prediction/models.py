import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import layers as layers
from typing import Tuple, Optional, Dict

# changing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps # numerical logs  

"""Based on the implementation of the Memory-Augmented Recurrent Neural Network (MRNN)
   from https://arxiv.org/pdf/2006.03860 and Variational Recurrent
   Neural Network (VRNN) from https://arxiv.org/abs/1506.02216."""

class RNN(nn.Module):
    """RNN model for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size)
        self.hidden2output = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_y, hidden_state=None):
        samples = input_y
        rnn_out, last_rnn_hidden = self.rnn(samples, hidden_state)
        output = self.hidden2output(rnn_out.view(-1, self.hidden_size))
        return output.view(samples.shape[0], samples.shape[1],
                           self.output_size), \
               last_rnn_hidden

class LSTM(nn.Module):
    """LSTM model for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.in_size = input_size
        self.h_size = hidden_size
        self.out_size = output_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.out_size)
        if hidden_state is None:
            h_0 = torch.zeros(batch_size, self.h_size)
            c_0 = torch.zeros(batch_size, self.h_size)
            hidden_state = (h_0, c_0)
        else:
            h_0 = hidden_state[0]
            c_0 = hidden_state[1]
        for times in range(time_steps):
            h_0, c_0 = self.lstm_cell(inputs[times, :], (h_0, c_0))
            outputs[times, :] = self.output(h_0)
        return outputs, (h_0, c_0)


class MRNNFixD(nn.Module):
    """mRNN with fixed d for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size, k, bias=True):
        super(MRNNFixD, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.b_d = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.mrnn_cell = layers.MRNNFixDCell(input_size, hidden_size,
                                             output_size, k)
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

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)
        self.d_matrix = 0.5 * F.sigmoid(self.b_d)
        weights_d = self.get_wd(self.d_matrix)
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        for times in range(time_steps):
            outputs[times, :], hidden_state = self.mrnn_cell(inputs[times, :], weights_d,
                                                   hidden_state)
        return outputs, hidden_state


class MRNN(nn.Module):
    """mRNN with dynamic d for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size, k, bias=True):
        super(MRNN, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.mrnn_cell = layers.MRNNCell(input_size, hidden_size,
                                         output_size, k)

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        for times in range(time_steps):
            outputs[times, :], hidden_state = self.mrnn_cell(inputs[times, :],
                                                             hidden_state)
        return outputs, hidden_state


class MLSTMFixD(nn.Module):
    """mLSTM with fixed d for time series prediction"""
    def __init__(self, input_size, hidden_size, k, output_size):
        super(MLSTMFixD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.d_values = Parameter(torch.Tensor(torch.zeros(1, hidden_size)),
                                  requires_grad=True)
        self.output_size = output_size
        self.mlstm_cell = layers.MLSTMFixDCell(self.input_size,
                                               self.hidden_size,
                                               self.output_size,
                                               self.k)
        self.sigmoid = nn.Sigmoid()

    def get_w(self, d_values):
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
                weights[:, sample, hidden] = self.get_w(d_value[0, hidden].
                                                        view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, hidden_states=None):
        if hidden_states is None:
            hidden = None
            h_c = None
        else:
            hidden = hidden_states[0]
            h_c = hidden_states[1]
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        self.d_values_sigmoid = 0.5 * F.sigmoid(self.d_values)
        weights_d = self.get_wd(self.d_values_sigmoid)
        for times in range(time_steps):
            outputs[times, :], hidden, h_c = self.mlstm_cell(inputs[times, :],
                                                             hidden,
                                                             h_c,
                                                             weights_d)
        return outputs, (hidden, h_c)


class MLSTM(nn.Module):
    """mLSTM with dynamic d for time series prediction"""
    def __init__(self, input_size, hidden_size, k, output_size):
        super(MLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.output_size = output_size
        self.mlstm_cell = layers.MLSTMCell(self.input_size, self.hidden_size,
                                           self.k, self.output_size)

    def forward(self, inputs, hidden_state=None):
        if hidden_state is None:
            hidden = None
            h_c = None
            d_values = None
        else:
            hidden = hidden_state[0]
            h_c = hidden_state[1]
            d_values = hidden_state[2]
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        for times in range(time_steps):
            outputs[times, :], hidden, h_c, d_values = \
                self.mlstm_cell(inputs[times, :], hidden, h_c, d_values)
        return outputs, (hidden, h_c, d_values)
    
################ Variational Recurrent Neural Network ####################

class VRNN(nn.Module):
    """VRNN model for time series prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, latent_size: int = None):
        super(VRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.latent_size = latent_size or hidden_size // 2
        
        # VRNN cell
        self.vrnn_cell = layers.VRNNCell(input_size, hidden_size, self.latent_size)
        
        # Output projection layer
        self.hidden2output = nn.Linear(self.hidden_size + self.latent_size, output_size)
        
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state"""
        return torch.zeros(batch_size, self.hidden_size, device=device)
    
    def forward(self, inputs: torch.Tensor, hidden_state: Optional[torch.Tensor] = None,
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
        self.nll_losses = []
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        
        for t in range(time_steps):
            # Get input at time t
            x_t = inputs[t, :]  # (batch_size, input_size)
            
            # VRNN cell forward pass
            z_t, hidden, info = self.vrnn_cell(x_t, hidden, sample)
            
            # Store KL and NLL losses
            self.kl_losses.append(info['kl_loss'])
            self.nll_losses.append(info['nll_loss'])
            
            # Generate output from hidden state and latent variable
            combined = torch.cat([hidden, z_t], dim=1)
            outputs[t, :] = self.hidden2output(combined)

        return outputs, hidden, info
    
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
    
################ Memory-Augmented Variational Recurrent Neural Network ####################

class MVRNNFixD(nn.Module):
    """mRNN with fixed d for time series prediction"""
    def __init__(self, input_size, hidden_size, latent_size, output_size, k, bias=True):
        super(MVRNNFixD, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.b_d = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.mvrnn_cell = layers.MVRNNFixDCell(input_size, hidden_size, latent_size,
                                             output_size, k)
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

    def forward(self, x, hidden_state=None, sample = True):
        time_steps = x.size(0)
        batch_size = x.size(1)
        self.d_matrix = 0.5 * F.sigmoid(self.b_d)
        weights_d = self.get_wd(self.d_matrix)
        self.kl_losses = []
        self.nll_losses = []

        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=x.dtype, device=x.device)
        for times in range(time_steps):
            outputs[times, :], hidden_state, info = self.mvrnn_cell(x[times, :], weights_d,
                                                   hidden_state, sample = sample)
            self.kl_losses.append(info['kl_loss'])
            self.nll_losses.append(info['nll_loss'])

        return outputs, hidden_state, info
    
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
    def __init__(self, input_size, hidden_size, output_size, latent_size, k, bias=False):
        super(MVRNN, self).__init__()

        self.x_dim = input_size
        self.h_dim = hidden_size
        self.output_size = output_size
        self.latent_size = latent_size

        self.mvrnn_cell = layers.MVRNNCell(self.x_dim, self.h_dim, output_size, k, latent_size)

        # self.hidden2output = nn.Linear(hidden_size + latent_size, output_size)

    def forward(self, x, hidden_state=None, sample=True):
        time_steps = x.size(0)
        batch_size = x.size(1)

        self.kl_losses = []
        self.nll_losses = []

        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=x.dtype, device=x.device)
        for times in range(time_steps):
            output, hidden_state, info = self.mvrnn_cell(x[times, :],
                                                             hidden_state, sample = sample)
            
            self.kl_losses.append(info['kl_loss'])
            self.nll_losses.append(info['nll_loss'])

            # combined = torch.cat([hidden_state[0], z_t], dim=1)
            outputs[times, :] = output

        return outputs, hidden_state, info

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
    """mRNN with fixed d for time series prediction"""
    def __init__(self, input_size, hidden_size, latent_size, output_size, k, bias=True):
        super(MVRNNFixD_WAE, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.b_d = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.mvrnn_cell_wae = layers.MVRNNFixDCell_WAE(input_size, hidden_size, latent_size,
                                             output_size, k)
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

    def forward(self, x, hidden_state=None, sample = True):
        time_steps = x.size(0)
        batch_size = x.size(1)
        self.d_matrix = 0.5 * F.sigmoid(self.b_d)
        weights_d = self.get_wd(self.d_matrix)
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=x.dtype, device=x.device)
        
        self.all_z_enc = []
        self.all_z_prior = []
        self.nll_losses = []

        for times in range(time_steps):
            outputs[times, :], hidden_state, info = self.mvrnn_cell_wae(x[times, :], weights_d,
                                                   hidden_state, sample = sample)
            self.all_z_enc.append(info['z_t'])
            self.all_z_prior.append(info['z_prior_t'])
            self.nll_losses.append(info['nll_loss'])

        info['all_z_enc'] = torch.stack(self.all_z_enc, dim=0)
        info['all_z_prior'] = torch.stack(self.all_z_prior, dim=0)

        return outputs, hidden_state, info
    
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
    def __init__(self, input_size, hidden_size, output_size, latent_size, k, bias=False):
        super(MVRNN_WAE, self).__init__()

        self.x_dim = input_size
        self.h_dim = hidden_size
        self.output_size = output_size
        self.latent_size = latent_size

        self.mvrnn_cell_wae = layers.MVRNNCell_WAE(self.x_dim, self.h_dim, output_size, k, latent_size)

    def forward(self, x, hidden_state=None, sample = True):
        time_steps = x.size(0)
        batch_size = x.size(1)

        self.all_z_enc = []
        self.all_z_prior = []
        self.nll_losses = []

        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=x.dtype, device=x.device)
        for times in range(time_steps):
            outputs[times, :], hidden_state, info = self.mvrnn_cell_wae(x[times, :],
                                                             hidden_state, sample = sample)
            
            self.all_z_enc.append(info['z_t'])
            self.all_z_prior.append(info['z_prior_t'])
            self.nll_losses.append(info['nll_loss'])

        info['all_z_enc'] = torch.stack(self.all_z_enc, dim=0)
        info['all_z_prior'] = torch.stack(self.all_z_prior, dim=0)

        return outputs, hidden_state, info

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