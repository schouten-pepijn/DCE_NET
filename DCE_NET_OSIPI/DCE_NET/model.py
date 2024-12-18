# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import functions
import matplotlib
matplotlib.use('TkAgg')

np.random.seed(42)
torch.manual_seed(42)

class DCE_NET(nn.Module):
    """
    network architecture
    """
    def __init__(self, hp):
        super(DCE_NET, self).__init__()
        self.hp = hp
        
        enc_dim = 0

        
        if self.hp.network.nn in ['lstm', 'gru']:
            if self.hp.network.nn == 'lstm':
                self.rnn = nn.LSTM(1, self.hp.network.layers[0], self.hp.network.layers[1], batch_first=True)
                hidden_dim = self.hp.network.layers[0]

            else:
                self.rnn = nn.GRU(1, self.hp.network.layers[0], self.hp.network.layers[1], batch_first=True)
                hidden_dim = self.hp.network.layers[0]

            if self.hp.network.attention:
                self.score_ke = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
                self.score_ve = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
                self.score_vp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
                self.score_dt = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))

                self.encoder_ke = nn.Sequential(nn.Linear(hidden_dim + enc_dim, int((hidden_dim + enc_dim)/2)),
                                                #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                                nn.ELU(),
                                                nn.Linear(int((hidden_dim + enc_dim)/2), 1)
                                                )
                self.encoder_ve = nn.Sequential(nn.Linear(hidden_dim + enc_dim, int((hidden_dim + enc_dim)/2)),
                                                #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                                nn.ELU(),
                                                nn.Linear(int((hidden_dim + enc_dim)/2), 1)
                                                )
                self.encoder_vp = nn.Sequential(nn.Linear(hidden_dim + enc_dim, int((hidden_dim + enc_dim)/2)),
                                                #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                                nn.ELU(),
                                                nn.Linear(int((hidden_dim + enc_dim)/2), 1)
                                                )
                self.encoder_dt = nn.Sequential(nn.Linear(hidden_dim + enc_dim, int((hidden_dim + enc_dim)/2)),
                                                #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                                nn.ELU(),
                                                nn.Linear(int((hidden_dim + enc_dim)/2), 1)
                                                )

            else:
                self.encoder = nn.Sequential(nn.Linear(hidden_dim + enc_dim, int((hidden_dim + enc_dim)/2)),
                                             #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                             nn.ELU(),
                                             nn.Linear(int((hidden_dim + enc_dim)/2), 4)
                                             )

    """
    forward pass for network training
    """
    def forward(self, X, aif_vals, first=False, epoch=0):
        if self.hp.network.nn in ['lstm', 'gru']:
            if self.hp.network.nn == 'lstm':
                output, (hn, cn) = self.rnn(X.unsqueeze(dim=2))
            else:
                output, hn = self.rnn(X.unsqueeze(dim=2))

            if self.hp.network.attention:
                score_ke = self.score_ke(output)
                score_ve = self.score_ve(output)
                score_vp = self.score_vp(output)
                score_dt = self.score_dt(output)
                
                hidden_ke = torch.sum(output*score_ke, dim=1)
                hidden_ve = torch.sum(output*score_ve, dim=1)
                hidden_vp = torch.sum(output*score_vp, dim=1)
                hidden_dt = torch.sum(output*score_dt, dim=1)

                ke = self.encoder_ke(hidden_ke).squeeze()
                ve = self.encoder_ve(hidden_ve).squeeze()
                vp = self.encoder_vp(hidden_vp).squeeze()
                dt = self.encoder_dt(hidden_dt).squeeze()

            else:
                hidden_enc = hn[-1]
                params = self.encoder(hidden_enc)

        ke_diff = self.hp.simulations.bounds[1, 0] - self.hp.simulations.bounds[0, 0]
        ve_diff = self.hp.simulations.bounds[1, 1] - self.hp.simulations.bounds[0, 1]
        vp_diff = self.hp.simulations.bounds[1, 2] - self.hp.simulations.bounds[0, 2]
        dt_diff = self.hp.simulations.bounds[1, 3] - self.hp.simulations.bounds[0, 3]

        if self.hp.network.attention:
            ke = self.hp.simulations.bounds[0, 0] + torch.sigmoid(ke.unsqueeze(1)) * ke_diff
            ve = self.hp.simulations.bounds[0, 1] + torch.sigmoid(ve.unsqueeze(1)) * ve_diff
            vp = self.hp.simulations.bounds[0, 2] + torch.sigmoid(vp.unsqueeze(1)) * vp_diff
            dt = self.hp.simulations.bounds[0, 3] + torch.sigmoid(dt.unsqueeze(1)) * dt_diff

        else:
            ke = self.hp.simulations.bounds[0, 0] + torch.sigmoid(params[:, 0].unsqueeze(1)) * ke_diff
            ve = self.hp.simulations.bounds[0, 1] + torch.sigmoid(params[:, 1].unsqueeze(1)) * ve_diff
            vp = self.hp.simulations.bounds[0, 2] + torch.sigmoid(params[:, 2].unsqueeze(1)) * vp_diff
            dt = self.hp.simulations.bounds[0, 3] + torch.sigmoid(params[:, 3].unsqueeze(1)) * dt_diff
        
        X_dw = functions.Cosine4AIF_ExtKety_pop(self.hp.acquisition.timing,
                                                aif_vals, ke, dt, ve, vp,
                                                self.hp.device)
        return X_dw, ke, dt, ve, vp


"""
optimizer for neural network
"""
def load_optimizer(net, hp):
    if hp.training.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=hp.training.lr, weight_decay=hp.training.weight_decay)
    elif hp.training.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=hp.training.lr, momentum=0.9, weight_decay=hp.training.weight_decay)
    elif hp.training.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=hp.training.lr, weight_decay=hp.training.weight_decay)
    else:
        raise Exception(
            'No valid optimiser is chosen. Please select a valid optimiser: training.optim = ''adam'', ''sgd'', ''adagrad''')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=hp.training.lr_mult,
                                                     patience=hp.training.optim_patience, verbose=True)

    return optimizer, scheduler
