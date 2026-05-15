import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Tuple, Dict

"""Based on the implementation of the Memory-Augmented Recurrent Neural Network (MRNN)
   from https://arxiv.org/pdf/2006.03860 and Variational Recurrent
   Neural Network (VRNN) from https://arxiv.org/abs/1506.02216."""

EPS = torch.finfo(torch.float).eps # numerical logs

class MRNNFixDCell(nn.RNNCellBase):
    """memory augmented RNN with fixed D"""
    def __init__(self, input_size, hidden_size, output_size, k, bias=True):
        super(MRNNFixDCell, self).__init__(input_size, hidden_size, bias,
                                           num_chunks=1)
        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.m_m = nn.Linear(hidden_size, hidden_size, bias=True)
        self.f_m = nn.Linear(input_size, hidden_size, bias=False)
        self.h_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.x_h = nn.Linear(input_size, hidden_size, bias=False)
        self.h_z = nn.Linear(hidden_size, output_size, bias=True)
        self.m_z = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, inputs, weight_d, hidden_state=None):
        if hidden_state is None:
            h_0 = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)
            m_0 = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)
            x_s = torch.zeros(self.k - 1, inputs.size(0), self.input_size,
                              dtype=inputs.dtype, device=inputs.device)
            hidden_state = (h_0, m_0, x_s)

        h_0 = hidden_state[0]
        m_0 = hidden_state[1]
        x_s = hidden_state[2]

        x_combine = torch.cat([x_s, inputs.view(-1, inputs.size(0),
                                         inputs.size(1))], 0)
        x_filter = torch.einsum('ijk,ik->ijk', [x_combine, weight_d]).\
            sum(dim=0)
        mem = F.tanh(self.m_m(m_0) + self.f_m(x_filter))
        hid = F.tanh(self.h_h(h_0) + self.x_h(inputs))
        z_out = F.tanh(self.h_z(hid) + self.m_z(mem))
        xs_out = x_combine[1:, :]
        return z_out, (hid, mem, xs_out)


class MRNNCell(nn.RNNCellBase):
    """memory augmented RNN with dynamic D"""
    def __init__(self, input_size, hidden_size, output_size, k, bias=True):
        super(MRNNCell, self).__init__(input_size, hidden_size, bias,
                                       num_chunks=1)  # weight_ih, weight_hh
        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.d_d = nn.Linear(input_size, input_size, bias=True)
        self.h_d = nn.Linear(hidden_size, input_size, bias=False)
        self.m_d = nn.Linear(hidden_size, input_size, bias=False)
        self.x_d = nn.Linear(input_size, input_size, bias=False)

        self.m_m = nn.Linear(hidden_size, hidden_size, bias=True)
        self.f_m = nn.Linear(input_size, hidden_size, bias=False)

        self.h_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.x_h = nn.Linear(input_size, hidden_size, bias=False)

        self.h_z = nn.Linear(hidden_size, output_size, bias=True)
        self.m_z = nn.Linear(hidden_size, output_size, bias=False)

    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def filter_d(self, h_c, d_values):
        weights = torch.ones(self.k, d_values.size(0), d_values.size(1))
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = \
                    self.get_ws(d_values[sample, hidden].view([1]))
        outputs = h_c.mul(weights).sum(dim=0)
        return outputs

    def forward(self, inputs, hidden_state=None):
        if hidden_state is None:
            h_0 = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)
            m_0 = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)
            d_0 = torch.zeros(inputs.size(0), self.input_size,
                              dtype=inputs.dtype, device=inputs.device)
            x_s = torch.zeros(self.k - 1, inputs.size(0), self.input_size,
                             dtype=inputs.dtype, device=inputs.device)
            hidden_state = (h_0, m_0, d_0, x_s)

        h_0 = hidden_state[0]
        m_0 = hidden_state[1]
        d_0 = hidden_state[2]
        x_s = hidden_state[3]

        # dynamic d
        d_values = 0.5 * F.sigmoid(self.d_d(d_0) + self.h_d(h_0) +
                                   self.m_d(m_0) + self.x_d(inputs))

        x_combine = torch.cat([x_s, inputs.view(-1, inputs.size(0),
                                         inputs.size(1))], 0)
        x_filter = self.filter_d(x_combine, d_values)
        mem = F.tanh(self.m_m(m_0) + self.f_m(x_filter))
        hid = F.tanh(self.h_h(h_0) + self.x_h(inputs))
        z_out = self.h_z(hid) + self.m_z(mem)
        xs_out = x_combine[1:, :]
        return z_out, (hid, mem, d_values, xs_out)


class MLSTMFixDCell(nn.Module):
    """memory augmented LSTM with fixed D"""
    def __init__(self, input_size, hidden_size, output_size, k):
        super(MLSTMFixDCell, self).__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.output_size = output_size

        self.c_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.i_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.f_gate = nn.Linear(input_size + 2 * hidden_size, hidden_size)
        self.o_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, sample, hidden, cell_tensor, weights):
        batch_size = sample.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size,
                                 dtype=sample.dtype, device=sample.device)
        if cell_tensor is None:
            cell_tensor = torch.zeros(self.k, batch_size, self.hidden_size,
                                      dtype=sample.dtype, device=sample.device)

        combined = torch.cat((sample, hidden), 1)
        first = torch.einsum('ijk,ik->ijk', [-cell_tensor, weights]).sum(dim=0)
        i_gate = self.i_gate(combined)
        o_gate = self.o_gate(combined)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.sigmoid(o_gate)
        c_tilde = self.c_gate(combined)
        c_tilde = self.tanh(c_tilde)

        second = torch.mul(c_tilde, i_gate)
        cell = torch.add(first, second)
        h_c = torch.cat([cell_tensor, cell.view([-1, cell.size(0),
                                                 cell.size(1)])], 0)
        h_c_1 = h_c[1:, :]
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.output(hidden)
        return output, hidden, h_c_1

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.hidden_size))


class MLSTMCell(nn.Module):
    """memory augmented LSTM with dynamic D"""
    def __init__(self, input_size, hidden_size, k, output_size):
        super(MLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.output_size = output_size

        self.c_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.i_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.f_gate = nn.Linear(input_size + 2 * hidden_size, hidden_size)
        self.o_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def get_ws(self, d_values):
        weights = [1.] * (self.k + 1)
        for i in range(0, self.k):
            weights[self.k - i - 1] = weights[self.k - i] * (i - d_values) / \
                                      (i + 1)
        return torch.cat(weights[0:self.k])

    def filter_d(self, cell_tensor, d_values):
        weights = torch.ones(self.k, d_values.size(0), d_values.size(1),
                       dtype=d_values.dtype, device=d_values.device)
        hidden_size = weights.shape[2]
        batch_size = weights.shape[1]
        for batch in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, batch, hidden] = \
                    self.get_ws(d_values[batch, hidden].view([1]))
        outputs = cell_tensor.mul(weights).sum(dim=0)
        return outputs

    def forward(self, sample, hidden, cell_tensor, d_0):
        batch_size = sample.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size,
                                 dtype=sample.dtype, device=sample.device)
        if cell_tensor is None:
            cell_tensor = torch.zeros(self.k, batch_size, self.hidden_size,
                                      dtype=sample.dtype, device=sample.device)
        if d_0 is None:
            d_0 = torch.zeros(batch_size, self.hidden_size,
                              dtype=sample.dtype, device=sample.device)

        combined = torch.cat((sample, hidden), 1)
        combined_d = torch.cat((sample, hidden, d_0), 1)
        d_values = self.f_gate(combined_d)
        d_values = self.sigmoid(d_values) * 0.5
        first = -self.filter_d(cell_tensor, d_values)
        i_gate = self.i_gate(combined)
        o_gate = self.o_gate(combined)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.sigmoid(o_gate)
        c_tilde = self.c_gate(combined)
        c_tilde = self.tanh(c_tilde)

        second = torch.mul(c_tilde, i_gate)
        cell = torch.add(first, second)
        h_c = torch.cat([cell_tensor, cell.view([-1, cell.size(0),
                                                 cell.size(1)])], 0)
        h_c_1 = h_c[1:, :]
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.output(hidden)
        return output, hidden, h_c_1, d_values

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.hidden_size))
    
################ Variational Recurrent Neural Network ####################
    
class VRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super(VRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Feature extraction for input

        self.phi_x = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Feature extraction for latent
        self.phi_z = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU()
        )

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(hidden_size, latent_size)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
        )

        # Prior: p(z_t | h_{t-1})
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(hidden_size, latent_size)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
        )

        # Decoder: p(x_t | z_t, h_{t-1})
        self.dec = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.decoder_mean = nn.Linear(hidden_size, input_size)  
        self.decoder_std = nn.Sequential(
            nn.Linear(hidden_size, input_size),  
            nn.Softplus()  # Ensures positive std
        )
        
        # Recurrence (h_t = f(phi_x_t, phi_z_t, h_{t-1}))
        self.rnn = nn.GRUCell(hidden_size + hidden_size, hidden_size) 
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Feature Extraction
        phi_x_t = self.phi_x(x)
        
        # Encoder q(z_t | x_≤t, h_{t-1})
        enc_input = torch.cat([phi_x_t, hidden], dim=1)
        enc_hidden = self.enc(enc_input)
        enc_mean = self.enc_mean(enc_hidden)
        enc_std = self.enc_std(enc_hidden)
        
        # Prior p(z_t | h_{t-1})
        prior_hidden = self.prior(hidden)
        prior_mean = self.prior_mean(prior_hidden)
        prior_std = self.prior_std(prior_hidden)
        
        # Sampling from encoder (during training) or prior (during generation)
        if sample:
            z_t = self._reparameterize_sample(enc_mean, enc_std, x.device)
        else:
            z_t = enc_mean
            
        phi_z_t = self.phi_z(z_t)
        
        # Decoder p(x_t | z_t, h_{t-1})
        dec_input = torch.cat([phi_z_t, hidden], dim=1)
        dec_hidden = self.dec(dec_input)
        dec_mean = self.decoder_mean(dec_hidden)
        dec_std = self.decoder_std(dec_hidden)
        
        # Recurrence
        rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1)
        new_hidden = self.rnn(rnn_input, hidden)
        
        # Computing KL divergence between encoder and prior
        kl_loss = self._kld_gauss(enc_mean, enc_std, prior_mean, prior_std)
        nll_loss = self._nll_gauss(dec_mean, dec_std, x)
        
        info = {
            'kl_loss': kl_loss,
            'nll_loss': nll_loss,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'enc_mean': enc_mean,  
            'enc_std': enc_std,   
            'dec_mean': dec_mean,
            'dec_std': dec_std
        }
        
        return z_t, new_hidden, info

    def _reparameterize_sample(self, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Reparameterization trick"""
        eps = torch.empty(size=std.size(), device=mean.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)

    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + (x - mean).pow(2)/(2*std.pow(2)))
    
################ Memory-Augmented Variational Recurrent Neural Network - VAE and WAE ####################

class MVRNNFixDCell(nn.RNNCellBase):
    """memory augmented RNN with fixed D"""
    def __init__(self, input_size, hidden_size, latent_size, output_size, k, bias=True):
        super(MVRNNFixDCell, self).__init__(input_size, hidden_size, bias,
                                           num_chunks=1)
        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.m_m = nn.Linear(hidden_size, hidden_size, bias=True)
        self.f_m = nn.Linear(input_size, hidden_size, bias=False)
        self.h_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.x_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_z = nn.Linear(hidden_size, output_size, bias=True)
        self.m_z = nn.Linear(hidden_size, output_size, bias=False)
        self.lat_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # # Feature extraction for input

        self.phi_x = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Feature extraction for latent
        self.phi_z = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU()
        )

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(hidden_size, latent_size)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
        )

        # Prior: p(z_t | h_{t-1})
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(hidden_size, latent_size)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
        )

        # Decoder: p(x_t | z_t, h_{t-1})
        self.dec = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.decoder_mean = nn.Linear(hidden_size, input_size)  
        self.decoder_std = nn.Sequential(
            nn.Linear(hidden_size, input_size),  
            nn.Softplus()  # Ensures positive std
        )

    def forward(self, x, weight_d, hidden_state=None, sample = True):
        if hidden_state is None:
            h_0 = torch.zeros(x.size(0), self.hidden_size,
                              dtype=x.dtype, device=x.device)
            m_0 = torch.zeros(x.size(0), self.hidden_size,
                              dtype=x.dtype, device=x.device)
            x_s = torch.zeros(self.k - 1, x.size(0), self.input_size,
                              dtype=x.dtype, device=x.device)
            hidden_state = (h_0, m_0, x_s)

        h_0 = hidden_state[0]
        m_0 = hidden_state[1]
        x_s = hidden_state[2]

        #Feature Extraction

        phi_x_t = self.phi_x(x)

        #Encoder
        encoder_input = self.enc(torch.cat([phi_x_t, h_0], dim=1))
        encoder_mean = self.enc_mean(encoder_input)
        encoder_std = self.enc_std(encoder_input)

        #Sampling
        z_t = self._reparameterize_sample(encoder_mean, encoder_std)
        phi_z_t = self.phi_z(z_t)

        #Prior
        prior_mean = self.prior_mean(h_0)
        prior_std = self.prior_std(h_0)

        #Decoder
        decoder_input = self.dec(torch.cat([phi_z_t, h_0], dim=1))
        decoder_mean = self.decoder_mean(decoder_input)
        decoder_std = self.decoder_std(decoder_input)

        #Recurrence 

        x_combine = torch.cat([x_s, x.view(-1, x.size(0),
                                         x.size(1))], 0)
        x_filter = torch.einsum('ijk,ik->ijk', [x_combine, weight_d]).\
            sum(dim=0)
        mem = F.tanh(self.m_m(m_0) + self.f_m(x_filter))
        new_hid = F.tanh(self.h_h(h_0) + self.x_h(phi_x_t) + self.lat_h(phi_z_t))
        z_out = F.tanh(self.h_z(new_hid) + self.m_z(mem))
        xs_out = x_combine[1:, :]

        #Computing KL divergence and Gauus Negative Log-Likelihood
        kl_loss = self._kld_gauss(encoder_mean, encoder_std, prior_mean, prior_std)
        nll_loss = self._nll_gauss(decoder_mean, decoder_std, x)

        info = {
            'kl_loss': kl_loss,
            'nll_loss': nll_loss,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'enc_mean': encoder_mean,
            'enc_std': encoder_std,
            'dec_mean': decoder_mean,
            'dec_std': decoder_std
        }

        return z_out, (new_hid, mem, xs_out), info
    
    def _reparameterize_sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        eps = torch.empty(size=std.size(), device=mean.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
            (std_1**2 + (mean_1 - mean_2)**2) /
            std_2**2 - 1)
        return	0.5 * torch.sum(kld_element)
    
    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + (x - mean).pow(2)/(2*std.pow(2)))

class MVRNNCell(nn.Module):
    """memory augmented RNN with dynamic D"""
    def __init__(self, input_size: int, hidden_size: int, 
                 output_size: int, k: int, latent_size: int = None):
        super(MVRNNCell, self).__init__()

        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size or hidden_size // 2
        self.output_size = output_size

        self.d_d = nn.Linear(input_size, input_size, bias=True)
        self.h_d = nn.Linear(hidden_size, input_size, bias=False)
        self.m_d = nn.Linear(hidden_size, input_size, bias=False)
        self.x_d = nn.Linear(input_size, input_size, bias=False)

        self.m_m = nn.Linear(hidden_size, hidden_size, bias=True)
        self.f_m = nn.Linear(input_size, hidden_size, bias=False)

        self.h_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.x_h = nn.Linear(hidden_size + hidden_size, hidden_size, bias=False)
        # self.lat_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.h_z = nn.Linear(hidden_size, output_size, bias=True)
        self.m_z = nn.Linear(hidden_size, output_size, bias=False)

        # Feature extraction for input
        self.phi_x = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Feature extraction for latent
        self.phi_z = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU()
        )

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(hidden_size, latent_size)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
        )

        # Prior: p(z_t | h_{t-1})
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(hidden_size, latent_size)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
        )

        # Decoder: p(x_t | z_t, h_{t-1})
        self.dec = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.decoder_mean = nn.Linear(hidden_size, input_size)  
        self.decoder_std = nn.Sequential(
            nn.Linear(hidden_size, input_size),  
            nn.Softplus()  # Ensures positive std
        )

    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def filter_d(self, h_c, d_values):
        weights = torch.ones(self.k, d_values.size(0), d_values.size(1))
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = \
                    self.get_ws(d_values[sample, hidden].view([1]))
        outputs = h_c.mul(weights).sum(dim=0)
        return outputs

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, sample=True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        if hidden is None:
            h_0 = torch.zeros(x.size(0), self.hidden_size,
                              dtype=x.dtype, device=x.device)
            m_0 = torch.zeros(x.size(0), self.hidden_size,
                              dtype=x.dtype, device=x.device)
            d_0 = torch.zeros(x.size(0), self.input_size,
                              dtype=x.dtype, device=x.device)
            x_s = torch.zeros(self.k - 1, x.size(0), self.input_size,
                              dtype=x.dtype, device=x.device)
            hidden = (h_0, m_0, d_0, x_s)

        h_0 = hidden[0]
        m_0 = hidden[1]
        d_0 = hidden[2]
        x_s = hidden[3]

        #Feature Extraction

        phi_x_t = self.phi_x(x)

        #Encoder
        encoder_input = self.enc(torch.cat([phi_x_t, h_0], dim=1))
        encoder_mean = self.enc_mean(encoder_input)
        encoder_std = self.enc_std(encoder_input)

        #Prior
        prior = self.prior(h_0)
        prior_mean = self.prior_mean(prior)
        prior_std = self.prior_std(prior)

        #Sampling
        if sample:
            z_t = self._reparameterize_sample(encoder_mean, encoder_std)
        else:
            z_t = encoder_mean
        phi_z_t = self.phi_z(z_t)

        #Decoder
        decoder_input = self.dec(torch.cat([phi_z_t, h_0], dim=1))
        decoder_mean = self.decoder_mean(decoder_input)
        decoder_std = self.decoder_std(decoder_input)

        #Recurrence

        # dynamic d
        d_values = 0.5 * F.sigmoid(self.d_d(d_0) + self.h_d(h_0) +
                                   self.m_d(m_0) + self.x_d(x))
        
        x_combine = torch.cat([x_s, x.view(-1, x.size(0),
                                         x.size(1))], 0)
        
        x_filter = self.filter_d(x_combine, d_values)
        mem = F.tanh(self.m_m(m_0) + self.f_m(x_filter))
        new_hid = F.tanh(self.h_h(h_0) + self.x_h(torch.cat([phi_x_t, phi_z_t], 1)))
        z_out = self.h_z(new_hid) + self.m_z(mem)
        xs_out = x_combine[1:, :]

        #Computing KL divergence and Gauss Negative Log-Likelihood
        kl_loss = self._kld_gauss(encoder_mean, encoder_std, prior_mean, prior_std)
        nll_loss = self._nll_gauss(decoder_mean, decoder_std, x)

        info = {
            'kl_loss': kl_loss,
            'nll_loss': nll_loss,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'enc_mean': encoder_mean,
            'enc_std': encoder_std,
            'dec_mean': decoder_mean,
            'dec_std': decoder_std
        }

        return z_out, (new_hid, mem, d_values, xs_out), info

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterize_sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        eps = torch.empty(size=std.size(), device=mean.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
            (std_1**2 + (mean_1 - mean_2)**2) /
            std_2**2 - 1)
        return	0.5 * torch.sum(kld_element)

    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + (x - mean).pow(2)/(2*std.pow(2)))
    
class MVRNNFixDCell_WAE(nn.RNNCellBase):
    """memory augmented RNN with fixed D"""
    def __init__(self, input_size, hidden_size, latent_size, output_size, k, bias=True):
        super(MVRNNFixDCell_WAE, self).__init__(input_size, hidden_size, bias,
                                           num_chunks=1)
        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.m_m = nn.Linear(hidden_size, hidden_size, bias=True)
        self.f_m = nn.Linear(input_size, hidden_size, bias=False)
        self.h_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.x_h = nn.Linear(hidden_size + hidden_size, hidden_size, bias=False)
        self.h_z = nn.Linear(hidden_size, output_size, bias=True)
        self.m_z = nn.Linear(hidden_size, output_size, bias=False)
        # self.lat_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # Feature extraction for input
        self.phi_x = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Feature extraction for latent
        self.phi_z = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU()
        )

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(hidden_size, latent_size)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
        )

        # Prior: p(z_t | h_{t-1})
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(hidden_size, latent_size)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
        )

        # Decoder: p(x_t | z_t, h_{t-1})
        self.dec = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.decoder_mean = nn.Linear(hidden_size, input_size)  
        self.decoder_std = nn.Sequential(
            nn.Linear(hidden_size, input_size),  
            nn.Softplus()  # Ensures positive std
        )

    def forward(self, x, weight_d, hidden_state=None, sample = True):
        if hidden_state is None:
            h_0 = torch.zeros(x.size(0), self.hidden_size,
                              dtype=x.dtype, device=x.device)
            m_0 = torch.zeros(x.size(0), self.hidden_size,
                              dtype=x.dtype, device=x.device)
            x_s = torch.zeros(self.k - 1, x.size(0), self.input_size,
                              dtype=x.dtype, device=x.device)
            hidden_state = (h_0, m_0, x_s)

        h_0 = hidden_state[0]
        m_0 = hidden_state[1]
        x_s = hidden_state[2]

        #Feature Extraction

        phi_x_t = self.phi_x(x)

        #Encoder
        encoder_input = self.enc(torch.cat([phi_x_t, h_0], dim=1))
        encoder_mean = self.enc_mean(encoder_input)
        encoder_std = self.enc_std(encoder_input) + 1e-3

        #Prior
        prior = self.prior(h_0)
        prior_mean = self.prior_mean(prior)
        prior_std = self.prior_std(prior) + 1e-3

        #Sampling
        if sample: 
            z_t = self._reparameterize_sample(encoder_mean, encoder_std)
            #Sampling from prior - MMD computations

            z_prior_t = self._reparameterize_sample(prior_mean, prior_std)

        else:
            z_t = encoder_mean

            z_prior_t = prior_mean
            
        #Feature Extraction
        phi_z_t = self.phi_z(z_t)

        #Decoder
        decoder_input = self.dec(torch.cat([phi_z_t, h_0], dim=1))
        decoder_mean = self.decoder_mean(decoder_input)
        decoder_std = self.decoder_std(decoder_input)

        #Computing Gauss Negative Log-Likelihood

        nll_loss = self._nll_gauss(decoder_mean, decoder_std, x)

        #Recurrence 

        x_combine = torch.cat([x_s, x.view(-1, x.size(0),
                                         x.size(1))], 0)
        x_filter = torch.einsum('ijk,ik->ijk', [x_combine, weight_d]).\
            sum(dim=0)
        mem = F.tanh(self.m_m(m_0) + self.f_m(x_filter))
        hid = F.tanh(self.h_h(h_0) + self.x_h(torch.cat([phi_x_t, phi_z_t], 1)))
        z_out = F.tanh(self.h_z(hid) + self.m_z(mem))
        xs_out = x_combine[1:, :]

        info = {
            'z_t': z_t,
            'z_prior_t': z_prior_t,
            'nll_loss': nll_loss,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'enc_mean': encoder_mean,
            'enc_std': encoder_std,
            'dec_mean': decoder_mean,
            'dec_std': decoder_std
        }

        return z_out, (hid, mem, xs_out), info
    
    def _reparameterize_sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        eps = torch.empty(size=std.size(), device=mean.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)
    
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
    
    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + (x - mean).pow(2)/(2*std.pow(2)))

class MVRNNCell_WAE(nn.Module):
    """memory augmented RNN with dynamic D"""
    def __init__(self, input_size: int, hidden_size: int, 
                 output_size: int, k: int, latent_size: int = None):
        super(MVRNNCell_WAE, self).__init__()

        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size or hidden_size // 2
        self.output_size = output_size

        self.d_d = nn.Linear(input_size, input_size, bias=True)
        self.h_d = nn.Linear(hidden_size, input_size, bias=False)
        self.m_d = nn.Linear(hidden_size, input_size, bias=False)
        self.x_d = nn.Linear(input_size, input_size, bias=False)

        self.m_m = nn.Linear(hidden_size, hidden_size, bias=True)
        self.f_m = nn.Linear(input_size, hidden_size, bias=False)

        self.h_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.x_h = nn.Linear(hidden_size + hidden_size, hidden_size, bias=False)
        # self.lat_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.h_z = nn.Linear(hidden_size, output_size, bias=True)
        self.m_z = nn.Linear(hidden_size, output_size, bias=False)
        
        self.phi_x = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Feature extraction for latent
        self.phi_z = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU()
        )

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(hidden_size, latent_size)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
        )

        # Prior: p(z_t | h_{t-1})
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(hidden_size, latent_size)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.Softplus()
        )

        # Decoder: p(x_t | z_t, h_{t-1})
        self.dec = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.decoder_mean = nn.Linear(hidden_size, input_size)  
        self.decoder_std = nn.Sequential(
            nn.Linear(hidden_size, input_size),  
            nn.Softplus()  # Ensures positive std
        )

    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def filter_d(self, h_c, d_values):
        weights = torch.ones(self.k, d_values.size(0), d_values.size(1))
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = \
                    self.get_ws(d_values[sample, hidden].view([1]))
        outputs = h_c.mul(weights).sum(dim=0)
        return outputs
    
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

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, sample = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        if hidden is None:
            h_0 = torch.zeros(x.size(0), self.hidden_size,
                              dtype=x.dtype, device=x.device)
            m_0 = torch.zeros(x.size(0), self.hidden_size,
                              dtype=x.dtype, device=x.device)
            d_0 = torch.zeros(x.size(0), self.input_size,
                              dtype=x.dtype, device=x.device)
            x_s = torch.zeros(self.k - 1, x.size(0), self.input_size,
                              dtype=x.dtype, device=x.device)
            hidden = (h_0, m_0, d_0, x_s)

        h_0 = hidden[0]
        m_0 = hidden[1]
        d_0 = hidden[2]
        x_s = hidden[3]

        #Feature Extraction

        phi_x_t = self.phi_x(x)

        #Encoder
        encoder_input = self.enc(torch.cat([phi_x_t, h_0], dim=1))
        encoder_mean = self.enc_mean(encoder_input)
        encoder_std = self.enc_std(encoder_input) + 1e-6

        #Prior
        prior = self.prior(h_0)
        prior_mean = self.prior_mean(prior)
        prior_std = self.prior_std(prior) + 1e-6

        #Sampling
        if sample: 
            z_t = self._reparameterize_sample(encoder_mean, encoder_std)
            #Sampling from prior - MMD computations

            z_prior_t = self._reparameterize_sample(prior_mean, prior_std) 

        else:
            z_t = encoder_mean

            z_prior_t = prior_mean 
            
        #Feature Extraction
        phi_z_t = self.phi_z(z_t)

        #Decoder
        decoder_input = self.dec(torch.cat([phi_z_t, h_0], dim=1))
        decoder_mean = self.decoder_mean(decoder_input)
        decoder_std = self.decoder_std(decoder_input) + 1e-6

        #Computing Gauss Negative Log-Likelihood

        nll_loss = self._nll_gauss(decoder_mean, decoder_std, x)

        #Recurrence

        # dynamic d
        d_values = 0.5 * torch.sigmoid(self.d_d(d_0) + self.h_d(h_0) +
                                   self.m_d(m_0) + self.x_d(x))
        
        x_combine = torch.cat([x_s, x.view(-1, x.size(0),
                                         x.size(1))], 0)
        
        x_filter = self.filter_d(x_combine, d_values)
        mem = torch.tanh(self.m_m(m_0) + self.f_m(x_filter))
        new_hid = torch.tanh(self.h_h(h_0) + self.x_h(torch.cat([phi_x_t, phi_z_t], 1)))
        z_out = self.h_z(new_hid) + self.m_z(mem)
        xs_out = x_combine[1:, :]

        info = {
            'z_t': z_t,
            'z_prior_t': z_prior_t,
            'nll_loss': nll_loss,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'enc_mean': encoder_mean,
            'enc_std': encoder_std,
            'dec_mean': decoder_mean,
            'dec_std': decoder_std
        }

        return z_out, (new_hid, mem, d_values, xs_out), info

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass

    def _reparameterize_sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        eps = torch.empty(size=std.size(), device=mean.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)

    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + (x - mean).pow(2)/(2*std.pow(2)))